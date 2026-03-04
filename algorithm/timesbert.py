from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TimesBERTConfig:
    num_variates: int
    lookback: int = 56
    horizon: int = 7
    patch_len: int = 7
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 128
    dropout: float = 0.1


class TimesBERT(nn.Module):
    """
    A lightweight TimesBERT-style model for multivariate time series.

    - Time series patch embedding (per variate)
    - Functional tokens ([CLS], [DOM], [MASK])
    - Encoder-only Transformer backbone
    - Forecast head for multi-step demand prediction
    - Two self-supervised objectives:
      1) masked patch modeling (MPM)
      2) variate discrimination (FTP-var)
    """

    def __init__(self, cfg: TimesBERTConfig) -> None:
        super().__init__()
        if cfg.lookback % cfg.patch_len != 0:
            raise ValueError("lookback must be divisible by patch_len")
        if cfg.num_variates <= 0:
            raise ValueError("num_variates must be > 0")

        self.cfg = cfg
        self.n_patch = cfg.lookback // cfg.patch_len

        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.var_embed = nn.Parameter(torch.randn(cfg.num_variates, cfg.d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(self.n_patch, cfg.d_model) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.dom_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.forecast_head = nn.Linear(cfg.d_model, cfg.horizon)
        self.mpm_head = nn.Linear(cfg.d_model, cfg.patch_len)
        self.var_cls_head = nn.Linear(cfg.d_model, cfg.num_variates)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, V] -> [B, V, N, P]
        b, l, v = x.shape
        if l != self.cfg.lookback:
            raise ValueError(f"expected lookback={self.cfg.lookback}, got {l}")
        if v != self.cfg.num_variates:
            raise ValueError(f"expected num_variates={self.cfg.num_variates}, got {v}")
        p = self.cfg.patch_len
        n = self.n_patch
        xt = x.reshape(b, n, p, v).permute(0, 3, 1, 2).contiguous()
        return xt

    def _encode_patches(self, patch_values: torch.Tensor, patch_embeds: torch.Tensor) -> torch.Tensor:
        # patch_embeds: [B, V, N, D] -> encoded patch tokens [B, V, N, D]
        b, v, n, d = patch_embeds.shape
        x = patch_embeds
        x = x + self.var_embed.view(1, v, 1, d) + self.pos_embed.view(1, 1, n, d)
        x = x.reshape(b, v * n, d)

        cls = self.cls_token.expand(b, -1, -1)
        dom = self.dom_token.expand(b, -1, -1)
        tokens = torch.cat([cls, dom, x], dim=1)
        h = self.encoder(tokens)
        patch_h = h[:, 2:, :].reshape(b, v, n, d)
        return patch_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_values = self._patchify(x)
        patch_embed = self.patch_embed(patch_values)
        patch_h = self._encode_patches(patch_values, patch_embed)

        # Aggregate patch-level tokens into variate-level representations.
        var_h = patch_h.mean(dim=2)  # [B, V, D]
        y = self.forecast_head(var_h)  # [B, V, H]
        return y.permute(0, 2, 1).contiguous()  # [B, H, V]

    def pretrain_objective(self, x: torch.Tensor, mask_ratio: float = 0.2) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return total pretraining loss and detached breakdown:
        - loss_mpm: masked patch reconstruction MSE
        - loss_ftp_var: variate discrimination CE
        """
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")

        patch_values = self._patchify(x)  # [B, V, N, P]
        patch_embed = self.patch_embed(patch_values)  # [B, V, N, D]

        b, v, n, d = patch_embed.shape
        mask = torch.rand(b, v, n, device=x.device) < mask_ratio
        mask_token = self.mask_token.view(1, 1, 1, d)
        patch_embed_masked = torch.where(mask.unsqueeze(-1), mask_token, patch_embed)

        patch_h = self._encode_patches(patch_values, patch_embed_masked)

        # 1) MPM
        patch_rec = self.mpm_head(patch_h)  # [B, V, N, P]
        mpm_diff = (patch_rec - patch_values) ** 2
        mpm_mask = mask.unsqueeze(-1).float()
        loss_mpm = (mpm_diff * mpm_mask).sum() / (mpm_mask.sum() * self.cfg.patch_len + 1e-8)

        # 2) FTP-var (predict variate id for each token)
        var_logits = self.var_cls_head(patch_h)  # [B, V, N, V]
        target_var = torch.arange(v, device=x.device).view(1, v, 1).expand(b, v, n)
        loss_ftp_var = nn.functional.cross_entropy(
            var_logits.reshape(b * v * n, v),
            target_var.reshape(b * v * n),
        )

        total = loss_mpm + 0.2 * loss_ftp_var
        parts = {
            "loss_mpm": loss_mpm.detach(),
            "loss_ftp_var": loss_ftp_var.detach(),
            "loss_total": total.detach(),
        }
        return total, parts

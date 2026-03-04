from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_manager import ensure_project_paths

ensure_project_paths()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="square,pad=0.01",
        linewidth=1.6,
        edgecolor="#222222",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, family="serif")


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            color="#222222",
        )
    )


def main() -> None:
    out_path = Path("figures/timesbert_model_structure.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=140)
    fig.patch.set_facecolor("#efefef")
    ax.set_facecolor("#efefef")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Parameter nodes (light blue)
    _add_box(ax, 0.08, 0.86, 0.16, 0.08, "var_embed\n[V, D]", "#b8d6e5")
    _add_box(ax, 0.30, 0.86, 0.16, 0.08, "pos_embed\n[N, D]", "#b8d6e5")
    _add_box(ax, 0.52, 0.86, 0.16, 0.08, "cls_token\n[1,1,D]", "#b8d6e5")
    _add_box(ax, 0.74, 0.86, 0.16, 0.08, "dom_token\n[1,1,D]", "#b8d6e5")

    # Backbone ops (light gray)
    _add_box(ax, 0.08, 0.72, 0.18, 0.08, "Input x\n[B,L,V]", "#e5e5e5")
    _add_box(ax, 0.31, 0.72, 0.18, 0.08, "_patchify\n[B,V,N,P]", "#e5e5e5")
    _add_box(ax, 0.54, 0.72, 0.18, 0.08, "patch_embed\nLinear(P,D)", "#e5e5e5")
    _add_box(ax, 0.77, 0.72, 0.16, 0.08, "Add tokens\n[CLS][DOM]", "#e5e5e5")

    _add_box(ax, 0.35, 0.58, 0.24, 0.08, "TransformerEncoder\n(batch_first=True)", "#e5e5e5")
    _add_box(ax, 0.35, 0.46, 0.24, 0.08, "patch_h\n[B,V,N,D]", "#e5e5e5")
    _add_box(ax, 0.35, 0.34, 0.24, 0.08, "mean(dim=2)\nvar_h [B,V,D]", "#e5e5e5")
    _add_box(ax, 0.35, 0.22, 0.24, 0.08, "forecast_head\nLinear(D,H)", "#e5e5e5")
    _add_box(ax, 0.35, 0.10, 0.24, 0.08, "Output y\n[B,H,V]", "#e5e5e5")

    # Pretrain branch
    _add_box(ax, 0.62, 0.58, 0.16, 0.08, "mask_token\n[1,1,1,D]", "#b8d6e5")
    _add_box(ax, 0.82, 0.58, 0.14, 0.08, "Mask patch\nratio", "#e5e5e5")
    _add_box(ax, 0.62, 0.44, 0.16, 0.08, "mpm_head\nLinear(D,P)", "#e5e5e5")
    _add_box(ax, 0.82, 0.44, 0.14, 0.08, "loss_mpm\nMSE", "#e5e5e5")
    _add_box(ax, 0.62, 0.30, 0.16, 0.08, "var_cls_head\nLinear(D,V)", "#e5e5e5")
    _add_box(ax, 0.82, 0.30, 0.14, 0.08, "loss_ftp_var\nCE", "#e5e5e5")
    _add_box(ax, 0.74, 0.16, 0.22, 0.08, "loss_total = loss_mpm + 0.2*loss_ftp_var", "#e5e5e5")

    # Arrows top-down and merges
    _arrow(ax, 0.17, 0.86, 0.17, 0.80)
    _arrow(ax, 0.39, 0.86, 0.39, 0.80)
    _arrow(ax, 0.61, 0.86, 0.78, 0.80)
    _arrow(ax, 0.83, 0.86, 0.85, 0.80)

    _arrow(ax, 0.26, 0.76, 0.31, 0.76)
    _arrow(ax, 0.49, 0.76, 0.54, 0.76)
    _arrow(ax, 0.72, 0.76, 0.77, 0.76)
    _arrow(ax, 0.85, 0.72, 0.47, 0.66)

    _arrow(ax, 0.47, 0.58, 0.47, 0.54)
    _arrow(ax, 0.47, 0.46, 0.47, 0.42)
    _arrow(ax, 0.47, 0.34, 0.47, 0.30)
    _arrow(ax, 0.47, 0.22, 0.47, 0.18)

    _arrow(ax, 0.69, 0.58, 0.82, 0.62)
    _arrow(ax, 0.89, 0.58, 0.89, 0.52)
    _arrow(ax, 0.47, 0.46, 0.62, 0.48)
    _arrow(ax, 0.78, 0.48, 0.82, 0.48)
    _arrow(ax, 0.47, 0.46, 0.62, 0.34)
    _arrow(ax, 0.78, 0.34, 0.82, 0.34)
    _arrow(ax, 0.89, 0.44, 0.85, 0.24)
    _arrow(ax, 0.89, 0.30, 0.85, 0.24)

    ax.text(0.03, 0.96, "TimesBERT Model Structure", fontsize=16, family="serif")
    ax.text(0.03, 0.93, "Style: autograd graph-like (parameter nodes in blue, ops in gray)", fontsize=11, family="serif")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

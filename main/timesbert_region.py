from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_manager import ensure_project_paths

ensure_project_paths()


@dataclass
class WindowedSplit:
    train_starts: list[int]
    val_starts: list[int]


class TimeWindowDataset:
    def __init__(self, series: np.ndarray, start_indices: list[int], lookback: int, horizon: int) -> None:
        self.series = series.astype(np.float32)
        self.start_indices = start_indices
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        import torch

        t = self.start_indices[idx]
        x = self.series[t - self.lookback : t, :]
        y = self.series[t : t + self.horizon, :]
        return torch.from_numpy(x), torch.from_numpy(y), t


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_region_panel(panel_path: Path) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    df = pd.read_csv(panel_path)
    required = {"region", "order_date", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in panel file: {sorted(missing)}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()

    pivot = (
        df.pivot_table(index="order_date", columns="region", values="demand", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )

    if pivot.empty:
        raise ValueError("region_day_panel is empty after parsing")

    dates = pd.DatetimeIndex(pivot.index)
    regions = [str(c) for c in pivot.columns]
    matrix = pivot.to_numpy(dtype=np.float32)
    return dates, regions, matrix


def _build_splits(timesteps: int, lookback: int, horizon: int, val_days: int) -> WindowedSplit:
    if timesteps <= lookback + horizon:
        raise ValueError("Not enough timesteps for requested lookback + horizon")
    if val_days <= horizon:
        raise ValueError("val_days must be > horizon")

    train_cut = timesteps - val_days
    all_starts = list(range(lookback, timesteps - horizon + 1))

    train_starts = [t for t in all_starts if t + horizon <= train_cut]
    val_starts = [t for t in all_starts if t >= train_cut and t + horizon <= timesteps]

    if not train_starts:
        raise ValueError("No train windows generated; reduce lookback/horizon/val_days")
    if not val_starts:
        raise ValueError("No validation windows generated; reduce horizon or val_days")

    return WindowedSplit(train_starts=train_starts, val_starts=val_starts)


def _denorm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return x * sigma.reshape(1, 1, -1) + mu.reshape(1, 1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch TimesBERT-style model on region/day panel.")
    parser.add_argument("--panel-path", type=Path, default=Path("data/processed/region_day_panel.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    parser.add_argument("--lookback", type=int, default=56)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--patch-len", type=int, default=7)
    parser.add_argument("--val-days", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    try:
        import torch
        from torch import nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader

        from algorithm.timesbert import TimesBERT, TimesBERTConfig
    except ModuleNotFoundError:
        print("PyTorch is not installed in current environment.")
        print("Please install torch first, then rerun: python3 main/run_timesbert_region.py")
        raise SystemExit(1)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    dates, regions, matrix = _load_region_panel(args.panel_path)
    t, v = matrix.shape
    split = _build_splits(t, args.lookback, args.horizon, args.val_days)

    train_cut = t - args.val_days
    mu = matrix[:train_cut, :].mean(axis=0)
    sigma = matrix[:train_cut, :].std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    matrix_norm = (matrix - mu.reshape(1, -1)) / sigma.reshape(1, -1)

    train_ds = TimeWindowDataset(matrix_norm, split.train_starts, args.lookback, args.horizon)
    val_ds = TimeWindowDataset(matrix_norm, split.val_starts, args.lookback, args.horizon)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TimesBERTConfig(
        num_variates=v,
        lookback=args.lookback,
        horizon=args.horizon,
        patch_len=args.patch_len,
        d_model=64,
        n_heads=4,
        n_layers=3,
        ff_dim=128,
        dropout=0.1,
    )
    model = TimesBERT(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    supervised_loss = nn.MSELoss()

    # Stage 1: self-supervised pretraining.
    model.train()
    for ep in range(args.pretrain_epochs):
        losses = []
        for xb, _, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = model.pretrain_objective(xb, mask_ratio=args.mask_ratio)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"[pretrain] epoch={ep + 1}/{args.pretrain_epochs} loss={np.mean(losses):.5f}")

    # Stage 2: supervised fine-tuning for horizon forecasting.
    best_state = None
    best_val = float("inf")
    for ep in range(args.finetune_epochs):
        model.train()
        train_losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = supervised_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(supervised_loss(pred, yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[finetune] epoch={ep + 1}/{args.finetune_epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Validation inference.
    model.eval()
    pred_chunks = []
    y_chunks = []
    t_chunks = []
    with torch.no_grad():
        for xb, yb, tb in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            pred_chunks.append(pred)
            y_chunks.append(yb.numpy())
            t_chunks.append(tb.numpy())

    pred_val_norm = np.concatenate(pred_chunks, axis=0)
    y_val_norm = np.concatenate(y_chunks, axis=0)
    t_val = np.concatenate(t_chunks, axis=0)

    pred_val = _denorm(pred_val_norm, mu, sigma)
    y_val = _denorm(y_val_norm, mu, sigma)

    y_true_flat = y_val.reshape(-1)
    y_pred_flat = pred_val.reshape(-1)

    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    mape = float(_mape(y_true_flat, y_pred_flat))

    # Save model.
    model_path = args.models_dir / "timesbert_region.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(cfg),
            "regions": regions,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
        },
        model_path,
    )

    # Save validation predictions in long format.
    rows = []
    for i, start_t in enumerate(t_val):
        for h in range(args.horizon):
            date_idx = int(start_t + h)
            for ridx, region in enumerate(regions):
                rows.append(
                    {
                        "region": region,
                        "order_date": dates[date_idx].date().isoformat(),
                        "window_start": dates[int(start_t)].date().isoformat(),
                        "horizon_step": int(h + 1),
                        "actual": float(y_val[i, h, ridx]),
                        "pred": float(pred_val[i, h, ridx]),
                    }
                )
    eval_pred_df = pd.DataFrame(rows)
    eval_pred_path = args.metrics_dir / "timesbert_eval_predictions.csv"
    eval_pred_df.to_csv(eval_pred_path, index=False)

    # Forecast next 7 days from latest lookback window.
    latest_x = matrix_norm[t - args.lookback : t, :]
    latest_x_t = torch.from_numpy(latest_x).unsqueeze(0).to(device)
    with torch.no_grad():
        future_norm = model(latest_x_t).cpu().numpy()
    future = _denorm(future_norm, mu, sigma)[0]  # [H, V]

    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=args.horizon, freq="D")
    future_rows = []
    for h in range(args.horizon):
        for ridx, region in enumerate(regions):
            future_rows.append(
                {
                    "region": region,
                    "order_date": future_dates[h].date().isoformat(),
                    "pred_demand": float(future[h, ridx]),
                }
            )
    future_df = pd.DataFrame(future_rows)
    future_path = args.predictions_dir / "region_daily_forecast_next_7_days_timesbert.csv"
    future_df.to_csv(future_path, index=False)

    metrics = {
        "model": "timesbert_region",
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "val_days": args.val_days,
        "num_regions": len(regions),
        "num_train_windows": len(split.train_starts),
        "num_val_windows": len(split.val_starts),
        "device": str(device),
        "panel_path": str(args.panel_path),
    }
    metrics_path = args.metrics_dir / "timesbert_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("TimesBERT run completed.")
    print(f"Data shape: T={t}, V={v}, regions={regions}")
    print(f"Train windows: {len(split.train_starts)} | Val windows: {len(split.val_starts)}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAPE(%): {mape:.2f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved eval predictions: {eval_pred_path}")
    print(f"Saved next-7-day forecast: {future_path}")


if __name__ == "__main__":
    main()

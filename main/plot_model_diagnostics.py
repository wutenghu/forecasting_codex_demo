from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_manager import ensure_project_paths

ensure_project_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model diagnostics from eval predictions.")
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=Path("artifacts/metrics/eval_predictions.csv"),
        help="Path of eval predictions csv.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures/model_diagnostics"),
        help="Output directory for diagnostic figures.",
    )
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred_path)
    df["order_date"] = pd.to_datetime(df["order_date"])

    # 1) Daily average trend: true vs predictions.
    daily = (
        df.groupby("order_date", as_index=False)[
            ["demand", "pred_naive_lag7", "pred_random_forest", "pred_xgboost"]
        ]
        .mean()
        .sort_values("order_date")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["order_date"], daily["demand"], label="actual", lw=2)
    ax.plot(daily["order_date"], daily["pred_naive_lag7"], label="naive_lag7", lw=1.5)
    ax.plot(daily["order_date"], daily["pred_random_forest"], label="random_forest", lw=1.5)
    ax.plot(daily["order_date"], daily["pred_xgboost"], label="xgboost", lw=1.5)
    ax.set_title("Daily Mean Demand: Actual vs Predicted")
    ax.set_xlabel("date")
    ax.set_ylabel("mean demand per region-day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.fig_dir / "daily_mean_actual_vs_pred.png", dpi=140)
    plt.close(fig)

    # 2) Naive scatter: actual vs naive prediction.
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = df.sample(n=min(10000, len(df)), random_state=42)
    ax.scatter(sample["demand"], sample["pred_naive_lag7"], s=8, alpha=0.2)
    max_v = float(max(sample["demand"].max(), sample["pred_naive_lag7"].max()))
    ax.plot([0, max_v], [0, max_v], "--", linewidth=1)
    ax.set_title("Naive Lag7: Actual vs Predicted")
    ax.set_xlabel("actual demand")
    ax.set_ylabel("predicted demand")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "naive_actual_vs_pred_scatter.png", dpi=140)
    plt.close(fig)

    # 3) Error distribution comparison.
    err = pd.DataFrame(
        {
            "naive_lag7": (df["pred_naive_lag7"] - df["demand"]).abs(),
            "random_forest": (df["pred_random_forest"] - df["demand"]).abs(),
            "xgboost": (df["pred_xgboost"] - df["demand"]).abs(),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in err.columns:
        err[c].clip(upper=3).plot(kind="kde", ax=ax, label=c)
    ax.set_title("Absolute Error Distribution (clipped at 3)")
    ax.set_xlabel("absolute error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.fig_dir / "abs_error_distribution.png", dpi=140)
    plt.close(fig)

    print(f"Saved figures to: {args.fig_dir}")
    print(" - daily_mean_actual_vs_pred.png")
    print(" - naive_actual_vs_pred_scatter.png")
    print(" - abs_error_distribution.png")


if __name__ == "__main__":
    main()

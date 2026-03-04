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


def _select_pred_column(df: pd.DataFrame) -> str:
    candidates = ["pred_xgboost", "pred_random_forest", "pred_naive_lag7"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No prediction column found in prediction file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top-N region actual vs prediction diagnostics.")
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=Path("artifacts/metrics/eval_predictions_tuned.csv"),
        help="Evaluation predictions file path.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures/model_diagnostics"),
        help="Output figure directory.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="Number of top regions by actual demand sum.",
    )
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred_path)
    df["order_date"] = pd.to_datetime(df["order_date"])

    pred_col = _select_pred_column(df)

    entity_col = "region"
    if entity_col not in df.columns:
        raise ValueError("Expected column `region` in prediction file for region-level diagnostics.")
    top_entities = (
        df.groupby(entity_col, as_index=False)["demand"]
        .sum()
        .sort_values("demand", ascending=False)
        .head(args.topn)[entity_col]
        .tolist()
    )

    summary = (
        df[df[entity_col].isin(top_entities)]
        .groupby(entity_col, as_index=False)
        .agg(actual_sum=("demand", "sum"), pred_sum=(pred_col, "sum"))
        .sort_values("actual_sum", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(args.fig_dir / "top_entities_summary.csv", index=False)

    for idx, entity in enumerate(top_entities, start=1):
        sub = df[df[entity_col] == entity].sort_values("order_date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub["order_date"], sub["demand"], label="actual", lw=2)
        ax.plot(sub["order_date"], sub[pred_col], label=pred_col, lw=1.8)
        ax.set_title(f"Top{idx:02d} {entity}: Actual vs {pred_col}")
        ax.set_xlabel("date")
        ax.set_ylabel("demand")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"top{idx:02d}_{entity}_actual_vs_pred.png", dpi=140)
        plt.close(fig)

    print(f"Prediction source: {args.pred_path}")
    print(f"Prediction column used: {pred_col}")
    print(f"Saved summary: {args.fig_dir / 'top_entities_summary.csv'}")
    print(f"Saved {len(top_entities)} figures to: {args.fig_dir}")


if __name__ == "__main__":
    main()

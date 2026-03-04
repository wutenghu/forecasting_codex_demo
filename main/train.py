from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_manager import ensure_project_paths

ensure_project_paths()


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _build_dataset(features_path: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()
    df["region_code"] = df["region"].astype("category").cat.codes.astype(int)
    return df.sort_values(["region", "order_date"], kind="mergesort").reset_index(drop=True)


def _select_feature_columns(df: pd.DataFrame, use_contemporaneous_shares: bool) -> list[str]:
    excluded = {"region", "order_date", "demand"}
    feature_cols = [c for c in df.columns if c not in excluded]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not use_contemporaneous_shares:
        numeric_cols = [
            c for c in numeric_cols if not (c.startswith("channel_share_") or c.startswith("drink_share_"))
        ]
    return numeric_cols


def _evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray, n_eval: int) -> dict[str, object]:
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(_mape(y_true, y_pred)),
        "n_eval": int(n_eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models and compare metrics.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/processed/region_day_features.csv"),
        help="Feature dataset path.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("artifacts/metrics"),
        help="Directory to save metric artifacts.",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=28,
        help="Validation horizon in trailing days.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--use-contemporaneous-shares",
        action="store_true",
        help="Include same-day channel/drink share features. Not recommended for true future forecasting.",
    )
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    df = _build_dataset(args.features_path)
    feature_cols = _select_feature_columns(df, args.use_contemporaneous_shares)
    max_date = df["order_date"].max()
    val_start = max_date - pd.Timedelta(days=args.val_days - 1)

    train_df = df[df["order_date"] < val_start].copy()
    val_df = df[df["order_date"] >= val_start].copy()

    # Ensure stable eval set for all models: valid target + lag_7 (naive baseline) + all feature columns.
    eval_mask = val_df["lag_7"].notna()
    for c in feature_cols:
        eval_mask &= val_df[c].notna()
    eval_df = val_df.loc[eval_mask].copy()

    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, "demand"]
    X_eval = eval_df[feature_cols]
    y_eval = eval_df["demand"].to_numpy()

    metrics: list[dict[str, object]] = []

    # A) Naive baseline: previous week same weekday.
    naive_pred = eval_df["lag_7"].to_numpy()
    metrics.append(_evaluate_predictions("naive_lag7", y_eval, naive_pred, len(eval_df)))
    naive_meta = {
        "type": "naive_baseline",
        "rule": "prediction = lag_7",
        "val_start": val_start.date().isoformat(),
        "val_end": max_date.date().isoformat(),
        "n_eval": int(len(eval_df)),
        "created_at": datetime.now(UTC).isoformat(),
    }
    with (args.models_dir / "naive_lag7.json").open("w", encoding="utf-8") as f:
        json.dump(naive_meta, f, ensure_ascii=False, indent=2)

    # B) RandomForest baseline.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=args.random_seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_eval)
    metrics.append(_evaluate_predictions("random_forest", y_eval, rf_pred, len(eval_df)))
    joblib.dump(rf, args.models_dir / "random_forest.joblib")

    # C) Candidate model: XGBoost.
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=args.random_seed,
        n_jobs=1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_eval)
    metrics.append(_evaluate_predictions("xgboost", y_eval, xgb_pred, len(eval_df)))
    joblib.dump(xgb, args.models_dir / "xgboost.joblib")

    comp = pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True)
    comp.to_csv(args.metrics_dir / "model_comparison.csv", index=False)
    eval_pred = eval_df[["region", "order_date", "demand"]].copy()
    eval_pred["pred_naive_lag7"] = naive_pred
    eval_pred["pred_random_forest"] = rf_pred
    eval_pred["pred_xgboost"] = xgb_pred
    eval_pred.to_csv(args.metrics_dir / "eval_predictions.csv", index=False)

    train_meta = {
        "features_path": str(args.features_path),
        "feature_count": len(feature_cols),
        "train_rows": int(len(X_train)),
        "eval_rows": int(len(eval_df)),
        "val_days": int(args.val_days),
        "use_contemporaneous_shares": bool(args.use_contemporaneous_shares),
        "val_start": val_start.date().isoformat(),
        "val_end": max_date.date().isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    with (args.metrics_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=2)

    print(f"Train rows: {len(X_train):,}")
    print(f"Eval rows: {len(eval_df):,}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Validation window: {val_start.date()} -> {max_date.date()}")
    print("Model comparison (sorted by RMSE):")
    print(comp.to_string(index=False))
    print(f"Saved metrics: {args.metrics_dir / 'model_comparison.csv'}")
    print(f"Saved eval predictions: {args.metrics_dir / 'eval_predictions.csv'}")
    print(f"Saved models: {args.models_dir}")


if __name__ == "__main__":
    main()

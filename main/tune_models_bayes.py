from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
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


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"region", "order_date", "demand"}
    cols = [c for c in df.columns if c not in excluded]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # Remove contemporaneous same-day share features for true forecasting setting.
    cols = [c for c in cols if not (c.startswith("channel_share_") or c.startswith("drink_share_"))]
    return cols


def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, n_eval: int) -> dict[str, object]:
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(_mape(y_true, y_pred)),
        "n_eval": int(n_eval),
    }


def _prepare_splits(df: pd.DataFrame, feature_cols: list[str], val_days: int, tune_days: int):
    max_date = df["order_date"].max()
    eval_start = max_date - pd.Timedelta(days=val_days - 1)

    pre_eval = df[df["order_date"] < eval_start].copy()
    eval_df = df[df["order_date"] >= eval_start].copy()

    tune_start = pre_eval["order_date"].max() - pd.Timedelta(days=tune_days - 1)
    tune_train = pre_eval[pre_eval["order_date"] < tune_start].copy()
    tune_valid = pre_eval[pre_eval["order_date"] >= tune_start].copy()

    # Ensure consistent non-null subset for each stage.
    mask_tune = tune_valid["lag_7"].notna()
    for c in feature_cols:
        mask_tune &= tune_valid[c].notna()
    tune_valid = tune_valid.loc[mask_tune].copy()

    mask_eval = eval_df["lag_7"].notna()
    for c in feature_cols:
        mask_eval &= eval_df[c].notna()
    eval_df = eval_df.loc[mask_eval].copy()

    X_tune_train = tune_train[feature_cols].dropna()
    y_tune_train = tune_train.loc[X_tune_train.index, "demand"]
    X_tune_valid = tune_valid[feature_cols]
    y_tune_valid = tune_valid["demand"].to_numpy()

    X_pre_eval = pre_eval[feature_cols].dropna()
    y_pre_eval = pre_eval.loc[X_pre_eval.index, "demand"]
    X_eval = eval_df[feature_cols]
    y_eval = eval_df["demand"].to_numpy()

    return {
        "max_date": max_date,
        "eval_start": eval_start,
        "tune_start": tune_start,
        "X_tune_train": X_tune_train,
        "y_tune_train": y_tune_train,
        "X_tune_valid": X_tune_valid,
        "y_tune_valid": y_tune_valid,
        "X_pre_eval": X_pre_eval,
        "y_pre_eval": y_pre_eval,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_df": eval_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization for RF/XGBoost and final retraining.")
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/region_day_features.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--val-days", type=int, default=28)
    parser.add_argument("--tune-days", type=int, default=28)
    parser.add_argument("--rf-trials", type=int, default=20)
    parser.add_argument("--xgb-trials", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    df = _build_dataset(args.features_path)
    feature_cols = _select_feature_columns(df)
    split = _prepare_splits(df, feature_cols, args.val_days, args.tune_days)

    X_tune_train = split["X_tune_train"]
    y_tune_train = split["y_tune_train"]
    X_tune_valid = split["X_tune_valid"]
    y_tune_valid = split["y_tune_valid"]
    X_pre_eval = split["X_pre_eval"]
    y_pre_eval = split["y_pre_eval"]
    X_eval = split["X_eval"]
    y_eval = split["y_eval"]
    eval_df = split["eval_df"]

    # ----- Bayesian tuning: RandomForest -----
    rf_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.random_seed))

    def rf_objective(trial: optuna.Trial) -> float:
        rf = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 80, 320),
            max_depth=trial.suggest_int("max_depth", 6, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 24),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            random_state=args.random_seed,
            n_jobs=-1,
        )
        rf.fit(X_tune_train, y_tune_train)
        pred = rf.predict(X_tune_valid)
        return float(np.sqrt(mean_squared_error(y_tune_valid, pred)))

    rf_study.optimize(rf_objective, n_trials=args.rf_trials, show_progress_bar=False)
    rf_best_params = rf_study.best_params
    rf_best = RandomForestRegressor(**rf_best_params, random_state=args.random_seed, n_jobs=-1)
    rf_best.fit(X_pre_eval, y_pre_eval)
    rf_pred = rf_best.predict(X_eval)

    # ----- Bayesian tuning: XGBoost -----
    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.random_seed))

    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "reg:squarederror",
            "random_state": args.random_seed,
            "n_jobs": 1,
        }
        model = XGBRegressor(**params)
        model.fit(
            X_tune_train,
            y_tune_train,
            eval_set=[(X_tune_valid, y_tune_valid)],
            verbose=False,
        )
        pred = model.predict(X_tune_valid)
        return float(np.sqrt(mean_squared_error(y_tune_valid, pred)))

    xgb_study.optimize(xgb_objective, n_trials=args.xgb_trials, show_progress_bar=False)
    xgb_best_params = xgb_study.best_params
    xgb_best = XGBRegressor(
        **xgb_best_params,
        objective="reg:squarederror",
        random_state=args.random_seed,
        n_jobs=1,
    )
    xgb_best.fit(X_pre_eval, y_pre_eval, eval_set=[(X_eval, y_eval)], verbose=False)
    xgb_pred = xgb_best.predict(X_eval)

    # Naive baseline for reference.
    naive_pred = eval_df["lag_7"].to_numpy()

    metrics = [
        _evaluate("naive_lag7", y_eval, naive_pred, len(y_eval)),
        _evaluate("random_forest_tuned", y_eval, rf_pred, len(y_eval)),
        _evaluate("xgboost_tuned", y_eval, xgb_pred, len(y_eval)),
    ]
    comp = pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True)
    comp.to_csv(args.metrics_dir / "model_comparison_tuned.csv", index=False)

    eval_pred = eval_df[["region", "order_date", "demand"]].copy()
    eval_pred["pred_naive_lag7"] = naive_pred
    eval_pred["pred_random_forest"] = rf_pred
    eval_pred["pred_xgboost"] = xgb_pred
    eval_pred.to_csv(args.metrics_dir / "eval_predictions_tuned.csv", index=False)

    joblib.dump(rf_best, args.models_dir / "random_forest_tuned.joblib")
    joblib.dump(xgb_best, args.models_dir / "xgboost_tuned.joblib")

    rf_trials_df = rf_study.trials_dataframe(attrs=("number", "value", "params", "state"))
    xgb_trials_df = xgb_study.trials_dataframe(attrs=("number", "value", "params", "state"))
    rf_trials_df.to_csv(args.metrics_dir / "rf_optuna_trials.csv", index=False)
    xgb_trials_df.to_csv(args.metrics_dir / "xgb_optuna_trials.csv", index=False)

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "features_path": str(args.features_path),
        "feature_count": len(feature_cols),
        "train_rows_for_tuning": int(len(X_tune_train)),
        "valid_rows_for_tuning": int(len(X_tune_valid)),
        "pre_eval_train_rows": int(len(X_pre_eval)),
        "eval_rows": int(len(X_eval)),
        "eval_start": split["eval_start"].date().isoformat(),
        "eval_end": split["max_date"].date().isoformat(),
        "rf_best_params": rf_best_params,
        "xgb_best_params": xgb_best_params,
        "rf_best_trial_rmse": float(rf_study.best_value),
        "xgb_best_trial_rmse": float(xgb_study.best_value),
    }
    with (args.metrics_dir / "tuning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Tuning train rows: {len(X_tune_train):,}, tuning valid rows: {len(X_tune_valid):,}")
    print(f"Final eval rows: {len(X_eval):,}")
    print("Best trial RMSE on tuning-valid:")
    print(f" - RF: {rf_study.best_value:.6f}")
    print(f" - XGB: {xgb_study.best_value:.6f}")
    print("Final eval model comparison (sorted by RMSE):")
    print(comp.to_string(index=False))
    print(f"Saved tuned metrics: {args.metrics_dir / 'model_comparison_tuned.csv'}")
    print(f"Saved tuned eval predictions: {args.metrics_dir / 'eval_predictions_tuned.csv'}")
    print(f"Saved tuned models: {args.models_dir}")


if __name__ == "__main__":
    main()

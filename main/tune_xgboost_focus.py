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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _build_dataset(features_path: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()
    df["region_code"] = df["region"].astype("category").cat.codes.astype(int)
    return df.sort_values(["region", "order_date"], kind="mergesort").reset_index(drop=True)


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"region", "order_date", "demand"}
    cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    # keep forecasting-safe setting (no same-day share leakage)
    cols = [c for c in cols if not (c.startswith("channel_share_") or c.startswith("drink_share_"))]
    return cols


def _prepare_splits(df: pd.DataFrame, feature_cols: list[str], val_days: int, tune_days: int):
    max_date = df["order_date"].max()
    eval_start = max_date - pd.Timedelta(days=val_days - 1)

    pre_eval = df[df["order_date"] < eval_start].copy()
    eval_df = df[df["order_date"] >= eval_start].copy()

    tune_start = pre_eval["order_date"].max() - pd.Timedelta(days=tune_days - 1)
    tune_train = pre_eval[pre_eval["order_date"] < tune_start].copy()
    tune_valid = pre_eval[pre_eval["order_date"] >= tune_start].copy()

    mask_tune = tune_valid["lag_7"].notna()
    for c in feature_cols:
        mask_tune &= tune_valid[c].notna()
    tune_valid = tune_valid.loc[mask_tune].copy()

    mask_eval = eval_df["lag_7"].notna()
    for c in feature_cols:
        mask_eval &= eval_df[c].notna()
    eval_df = eval_df.loc[mask_eval].copy()

    x_tune_train = tune_train[feature_cols].dropna()
    y_tune_train = tune_train.loc[x_tune_train.index, "demand"]
    x_tune_valid = tune_valid[feature_cols]
    y_tune_valid = tune_valid["demand"].to_numpy()

    x_pre_eval = pre_eval[feature_cols].dropna()
    y_pre_eval = pre_eval.loc[x_pre_eval.index, "demand"]
    x_eval = eval_df[feature_cols]
    y_eval = eval_df["demand"].to_numpy()

    return {
        "max_date": max_date,
        "eval_start": eval_start,
        "x_tune_train": x_tune_train,
        "y_tune_train": y_tune_train,
        "x_tune_valid": x_tune_valid,
        "y_tune_valid": y_tune_valid,
        "x_pre_eval": x_pre_eval,
        "y_pre_eval": y_pre_eval,
        "x_eval": x_eval,
        "y_eval": y_eval,
        "eval_df": eval_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused Bayesian tuning for XGBoost.")
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/region_day_features.csv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--val-days", type=int, default=28)
    parser.add_argument("--tune-days", type=int, default=28)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    df = _build_dataset(args.features_path)
    feature_cols = _select_feature_columns(df)
    split = _prepare_splits(df, feature_cols, args.val_days, args.tune_days)

    x_tune_train = split["x_tune_train"]
    y_tune_train = split["y_tune_train"]
    x_tune_valid = split["x_tune_valid"]
    y_tune_valid = split["y_tune_valid"]
    x_pre_eval = split["x_pre_eval"]
    y_pre_eval = split["y_pre_eval"]
    x_eval = split["x_eval"]
    y_eval = split["y_eval"]
    eval_df = split["eval_df"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.random_seed))

    def objective(trial: optuna.Trial) -> float:
        objective_name = trial.suggest_categorical(
            "objective", ["reg:squarederror", "reg:pseudohubererror", "count:poisson"]
        )
        params = {
            "objective": objective_name,
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "random_state": args.random_seed,
            "n_jobs": 1,
        }
        model = XGBRegressor(**params)
        model.fit(x_tune_train, y_tune_train, eval_set=[(x_tune_valid, y_tune_valid)], verbose=False)
        pred = np.maximum(0.0, model.predict(x_tune_valid))
        return _rmse(y_tune_valid, pred)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best_params = study.best_params

    best_model = XGBRegressor(
        **best_params,
        tree_method="hist",
        random_state=args.random_seed,
        n_jobs=1,
    )
    best_model.fit(x_pre_eval, y_pre_eval, eval_set=[(x_eval, y_eval)], verbose=False)
    pred_eval = np.maximum(0.0, best_model.predict(x_eval))

    metrics = {
        "model": "xgboost_tuned_focus",
        "mae": float(mean_absolute_error(y_eval, pred_eval)),
        "rmse": _rmse(y_eval, pred_eval),
        "mape_pct": _mape(y_eval, pred_eval),
        "n_eval": int(len(y_eval)),
    }

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(args.metrics_dir / "xgb_optuna_trials_focus.csv", index=False)

    eval_pred = eval_df[["region", "order_date", "demand"]].copy()
    eval_pred["pred_xgboost_focus"] = pred_eval
    eval_pred.to_csv(args.metrics_dir / "eval_predictions_xgb_focus.csv", index=False)

    with (args.metrics_dir / "xgb_focus_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "features_path": str(args.features_path),
                "feature_count": len(feature_cols),
                "trials": args.trials,
                "val_start": split["eval_start"].date().isoformat(),
                "val_end": split["max_date"].date().isoformat(),
                "best_trial_rmse_tune_valid": float(study.best_value),
                "best_params": best_params,
                "eval_metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(best_model, args.models_dir / "xgboost_tuned_focus.joblib")

    print(f"Trials: {args.trials}")
    print(f"Tune-valid best RMSE: {study.best_value:.6f}")
    print(
        f"Final eval - MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, MAPE: {metrics['mape_pct']:.6f}"
    )
    print(f"Saved: {args.metrics_dir / 'xgb_focus_metrics.json'}")


if __name__ == "__main__":
    main()

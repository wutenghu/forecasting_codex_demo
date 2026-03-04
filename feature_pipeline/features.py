from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


@dataclass(frozen=True)
class FeatureEngineeringPipelineConfig:
    input_path: Path = Path("data/processed/region_day_panel.csv")
    raw_data_path: Path = Path("data/starbucks_customer_ordering_patterns.csv")
    output_path: Path = Path("data/processed/region_day_features.csv")
    random_seed: int = 42


class FeatureEngineeringPipeline:
    def __init__(self, config: FeatureEngineeringPipelineConfig) -> None:
        self.config = config

    def run(self) -> dict[str, object]:
        # Keep seed in config for reproducibility governance.
        _ = self.config.random_seed

        df = self._load_panel(self.config.input_path)
        raw_context = self._load_raw_context(self.config.raw_data_path)
        df = self._add_calendar_features(df)
        df = self._add_channel_share_features(df, raw_context)
        df = self._add_drink_share_features(df, raw_context)
        df = self._add_holiday_features(df, raw_context)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_region_stat_features(df)
        self._save_features(df, self.config.output_path)

        summary = {
            "input_rows": len(df),
            "regions": int(df["region"].nunique()),
            "date_min": df["order_date"].min().date().isoformat(),
            "date_max": df["order_date"].max().date().isoformat(),
            "output_rows": len(df),
            "output_cols": len(df.columns),
            "output_path": str(self.config.output_path),
        }
        return summary

    @staticmethod
    def _load_panel(input_path: Path) -> pd.DataFrame:
        df = pd.read_csv(input_path)
        required = {"region", "order_date", "demand"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df = df.dropna(subset=["order_date"]).copy()
        return df.sort_values(["region", "order_date"], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _load_raw_context(raw_data_path: Path) -> pd.DataFrame:
        cols = ["region", "order_date", "order_channel", "drink_category"]
        raw = pd.read_csv(raw_data_path, usecols=cols)
        raw["order_date"] = pd.to_datetime(raw["order_date"], errors="coerce")
        raw = raw.dropna(subset=["order_date"]).copy()
        return raw

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["dow"] = out["order_date"].dt.dayofweek
        out["day"] = out["order_date"].dt.day
        out["month"] = out["order_date"].dt.month
        out["weekofyear"] = out["order_date"].dt.isocalendar().week.astype(int)
        out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
        return out

    @staticmethod
    def _add_channel_share_features(df: pd.DataFrame, raw_context: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        grp = (
            raw_context.groupby(["region", "order_date", "order_channel"], as_index=False)
            .size()
            .rename(columns={"size": "cnt"})
        )
        totals = grp.groupby(["region", "order_date"], as_index=False)["cnt"].sum().rename(columns={"cnt": "total"})
        grp = grp.merge(totals, on=["region", "order_date"], how="left")
        grp["share"] = grp["cnt"] / grp["total"]

        pivot = grp.pivot_table(
            index=["region", "order_date"],
            columns="order_channel",
            values="share",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        pivot.columns = [
            str(c) if c in ["region", "order_date"] else f"channel_share_{str(c).strip().lower().replace(' ', '_').replace('-', '_')}"
            for c in pivot.columns
        ]
        out = base.merge(pivot, on=["region", "order_date"], how="left")
        channel_cols = [c for c in out.columns if c.startswith("channel_share_")]
        out[channel_cols] = out[channel_cols].fillna(0.0)
        return out

    @staticmethod
    def _add_drink_share_features(df: pd.DataFrame, raw_context: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        grp = (
            raw_context.groupby(["region", "order_date", "drink_category"], as_index=False)
            .size()
            .rename(columns={"size": "cnt"})
        )
        totals = grp.groupby(["region", "order_date"], as_index=False)["cnt"].sum().rename(columns={"cnt": "total"})
        grp = grp.merge(totals, on=["region", "order_date"], how="left")
        grp["share"] = grp["cnt"] / grp["total"]

        pivot = grp.pivot_table(
            index=["region", "order_date"],
            columns="drink_category",
            values="share",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        pivot.columns = [
            str(c) if c in ["region", "order_date"] else f"drink_share_{str(c).strip().lower().replace(' ', '_').replace('-', '_')}"
            for c in pivot.columns
        ]
        out = base.merge(pivot, on=["region", "order_date"], how="left")
        drink_cols = [c for c in out.columns if c.startswith("drink_share_")]
        out[drink_cols] = out[drink_cols].fillna(0.0)
        return out

    @staticmethod
    def _add_holiday_features(df: pd.DataFrame, raw_context: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=out["order_date"].min(), end=out["order_date"].max())
        out["is_us_federal_holiday"] = out["order_date"].isin(holidays).astype(int)
        for region in sorted(out["region"].dropna().unique()):
            safe = str(region).strip().lower().replace(" ", "_").replace("-", "_")
            out[f"is_region_{safe}"] = (out["region"] == region).astype(int)
            out[f"is_holiday_region_{safe}"] = (
                ((out["region"] == region) & (out["is_us_federal_holiday"] == 1)).astype(int)
            )
        return out

    @staticmethod
    def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        g = out.groupby("region")["demand"]
        for lag in (1, 7, 14, 28):
            out[f"lag_{lag}"] = g.shift(lag)
        return out

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        g = out.groupby("region")["demand"]
        out["rolling_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
        out["rolling_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
        return out

    @staticmethod
    def _add_region_stat_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        stats = out.groupby("region")["demand"].agg(["mean", "std"]).fillna(0.0)
        stats.columns = ["region_avg_demand", "region_std_demand"]
        out = out.merge(stats, on="region", how="left")
        return out

    @staticmethod
    def _save_features(df: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

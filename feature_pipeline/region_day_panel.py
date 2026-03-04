from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RegionDayPanelPipelineConfig:
    data_path: Path = Path("data/starbucks_customer_ordering_patterns.csv")
    output_path: Path = Path("data/processed/region_day_panel.csv")
    random_seed: int = 42


class RegionDayPanelPipeline:
    def __init__(self, config: RegionDayPanelPipelineConfig) -> None:
        self.config = config

    def run(self) -> dict[str, object]:
        _ = self.config.random_seed

        raw = self._load_and_validate(self.config.data_path)
        panel = self._build_region_day_panel(raw)
        self._save_panel(panel, self.config.output_path)

        active_ratio = (panel["demand"] > 0).mean() * 100
        return {
            "input_rows": len(raw),
            "regions": int(panel["region"].nunique()),
            "date_min": panel["order_date"].min().date().isoformat(),
            "date_max": panel["order_date"].max().date().isoformat(),
            "output_rows": len(panel),
            "active_region_day_ratio_pct": round(active_ratio, 2),
            "output_path": str(self.config.output_path),
        }

    @staticmethod
    def _load_and_validate(data_path: Path) -> pd.DataFrame:
        df = pd.read_csv(data_path, usecols=["region", "order_date", "order_id"])
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df = df.dropna(subset=["order_date"]).copy()
        return df.sort_values(["region", "order_date", "order_id"], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _build_region_day_panel(df: pd.DataFrame) -> pd.DataFrame:
        daily = (
            df.groupby(["region", "order_date"], as_index=False)
            .size()
            .rename(columns={"size": "demand"})
        )
        all_regions = sorted(daily["region"].unique())
        all_dates = pd.date_range(daily["order_date"].min(), daily["order_date"].max(), freq="D")
        full_idx = pd.MultiIndex.from_product([all_regions, all_dates], names=["region", "order_date"])
        panel = (
            daily.set_index(["region", "order_date"])
            .reindex(full_idx, fill_value=0)
            .reset_index()
        )
        panel["demand"] = panel["demand"].astype(int)
        return panel.sort_values(["region", "order_date"], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _save_panel(panel: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_manager import ensure_project_paths

ensure_project_paths()

from feature_pipeline.features import FeatureEngineeringPipeline, FeatureEngineeringPipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible region-day feature dataset.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/region_day_panel.csv"),
        help="Input region-day panel path.",
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=Path("data/starbucks_customer_ordering_patterns.csv"),
        help="Raw order-level data path for context aggregation features.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/region_day_features.csv"),
        help="Output feature dataset path.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed value kept in config for reproducibility governance.",
    )
    args = parser.parse_args()

    config = FeatureEngineeringPipelineConfig(
        input_path=args.input_path,
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        random_seed=args.random_seed,
    )
    summary = FeatureEngineeringPipeline(config).run()

    print(f"Input rows: {summary['input_rows']:,}")
    print(f"Regions: {summary['regions']}")
    print(f"Date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"Output rows: {summary['output_rows']:,}")
    print(f"Output columns: {summary['output_cols']}")
    print(f"Saved: {summary['output_path']}")


if __name__ == "__main__":
    main()

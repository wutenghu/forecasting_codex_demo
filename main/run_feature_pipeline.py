from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_manager import ensure_project_paths

ensure_project_paths()

from main.feature_pipeline import main


if __name__ == "__main__":
    main()

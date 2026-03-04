from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_ROOT / "main"


def ensure_project_paths() -> None:
    project_root_str = str(PROJECT_ROOT)
    main_dir_str = str(MAIN_DIR)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if main_dir_str not in sys.path:
        sys.path.insert(0, main_dir_str)

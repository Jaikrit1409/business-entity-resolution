"""Shared helpers for paths, configuration, and reproducibility."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


def set_random_seed(seed: int = 42) -> None:
    """Set Python and NumPy random seeds when NumPy is available."""
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def ensure_directories() -> None:
    for path in (MODEL_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)

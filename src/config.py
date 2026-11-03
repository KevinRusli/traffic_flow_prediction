"""Load the project configuration (config.yaml) as a nested dict.

Usage:
    from src.config import load_config, PROJECT_ROOT
    cfg = load_config()
    k = cfg["routing"]["k_paths"]
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=4)
def _load_cached(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the configuration dict. A fresh copy is returned so callers can mutate it."""
    import copy

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    return copy.deepcopy(_load_cached(str(cfg_path)))


def resolve_path(relative: str | Path) -> Path:
    """Turn a path from config.yaml (relative to the repo root) into an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p

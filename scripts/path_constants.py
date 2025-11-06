"""Shared path utilities and defaults for analysis scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

DEFAULT_TABLES_DIR = "tables"
DEFAULT_FIG_DIRS = ["fig", "figures"]
DEFAULT_OUTPUT_DIR = "output"


def ensure_dir(path: Path | str) -> Path:
    """Ensure *path* exists as a directory and return it as a :class:`Path`."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def first_existing(paths: Iterable[Path | str]) -> Optional[Path]:
    """Return the first existing path from *paths*, or ``None`` if none exist."""
    for candidate in paths:
        cand_path = Path(candidate)
        if cand_path.exists():
            return cand_path
    return None


def make_parent(path: Path | str) -> Path:
    """Ensure the parent directory of *path* exists and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

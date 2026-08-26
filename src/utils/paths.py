"""Path helpers for DataPilot AI."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a path relative to project root unless absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or project_root()
    return (root / p).resolve()


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if missing; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_path(name: str, default: str) -> Path:
    """Read a path from environment or fall back to ``default`` (project-relative)."""
    value = os.getenv(name, default)
    return resolve_path(value)

"""Resolve the trained LoRA adapter for System C (FT + RAG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import load_yaml
from src.utils.paths import resolve_path

ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
DEFAULT_RUN_ADAPTER = "colab_t4_qlora_v1/adapter"


def is_lora_adapter_dir(path: Path) -> bool:
    """True if the directory looks like a PEFT adapter (config + weights)."""
    if not path.is_dir():
        return False
    if not (path / "adapter_config.json").is_file():
        return False
    return any((path / name).is_file() for name in ADAPTER_WEIGHT_NAMES)


def _as_adapter_dir(path: Path) -> Path | None:
    if is_lora_adapter_dir(path):
        return path
    nested = path / "adapter"
    if is_lora_adapter_dir(nested):
        return nested
    return None


def resolve_lora_adapter_path(
    model_cfg: dict[str, Any] | None = None,
    *,
    explicit: str | Path | None = None,
) -> Path | None:
    """Return the adapter directory to load, or None if none is available.

    Order: explicit path → ``paths.adapter_path`` in model.yaml →
    ``colab_t4_qlora_v1/adapter`` → newest ``*/adapter`` under adapter_dir.
    """
    if explicit:
        found = _as_adapter_dir(resolve_path(explicit))
        if found is not None:
            return found

    cfg = model_cfg if model_cfg is not None else load_yaml("./config/model.yaml")
    paths = cfg.get("paths", {})
    configured = paths.get("adapter_path")
    if configured:
        found = _as_adapter_dir(resolve_path(str(configured)))
        if found is not None:
            return found

    base = resolve_path(str(paths.get("adapter_dir", "./experiments/results/adapters")))
    preferred = base / DEFAULT_RUN_ADAPTER
    found = _as_adapter_dir(preferred)
    if found is not None:
        return found

    if not base.is_dir():
        return None
    candidates: list[Path] = []
    for child in base.iterdir():
        adapter = _as_adapter_dir(child)
        if adapter is not None:
            candidates.append(adapter)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.paths import project_root, resolve_path


def load_env(env_file: str | Path | None = None) -> None:
    """Load ``.env`` from project root (if present). Does not override existing env vars."""
    path = resolve_path(env_file) if env_file else project_root() / ".env"
    if path.exists():
        load_dotenv(path, override=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {resolved}")
    return data


def load_sources_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load curated source inventory from ``config/sources.yaml``."""
    load_env()
    config_dir = os.getenv("DATAPILOT_CONFIG_DIR", "./config")
    default = resolve_path(config_dir) / "sources.yaml"
    return load_yaml(path or default)


def get_user_agent(sources_cfg: dict[str, Any] | None = None) -> str:
    """Resolve HTTP User-Agent from env or polite default."""
    load_env()
    env_name = "DATAPILOT_USER_AGENT"
    if sources_cfg:
        env_name = (
            sources_cfg.get("politeness", {}).get("user_agent_env")
            or env_name
        )
    return os.getenv(
        env_name,
        "DataPilotAI-AcademicResearch/0.1 (Masters Project; respectful crawler)",
    )


def get_request_delay_seconds(sources_cfg: dict[str, Any]) -> float:
    """Resolve inter-request delay from env override or sources politeness config."""
    load_env()
    env_val = os.getenv("DATAPILOT_REQUEST_DELAY_SECONDS")
    if env_val is not None and env_val.strip() != "":
        return float(env_val)
    return float(sources_cfg.get("politeness", {}).get("default_delay_seconds", 1.5))

"""Unit tests for ingestion helpers (no network / no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.inventory import list_topic_targets
from src.ingestion.models import TopicTarget
from src.ingestion.robots import RobotsCache
from src.utils.config import load_sources_config


def test_sources_config_loads_and_has_expected_counts() -> None:
    cfg = load_sources_config()
    targets = list_topic_targets(cfg)
    assert cfg["inventory_version"] == "0.1.2"
    assert len(cfg["sources"]) == 6
    assert len(targets) == 42
    assert all(isinstance(t, TopicTarget) for t in targets)
    assert all(t.url.startswith("http") for t in targets)


def test_document_ids_are_unique() -> None:
    cfg = load_sources_config()
    targets = list_topic_targets(cfg)
    ids = [t.document_id for t in targets]
    assert len(ids) == len(set(ids))


def test_topic_target_document_id() -> None:
    target = TopicTarget(
        source_id="postgresql",
        source_name="PostgreSQL Documentation",
        category="SQL",
        license_note="note",
        topic_id="pg_select",
        title="SELECT",
        url="https://www.postgresql.org/docs/current/sql-select.html",
        topic="SELECT",
    )
    assert target.document_id == "postgresql__pg_select"


def test_robots_cache_uses_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = RobotsCache(user_agent="DataPilotAI-Test/0.1")

    class FakeParser:
        def can_fetch(self, agent: str, url: str) -> bool:
            assert agent == "DataPilotAI-Test/0.1"
            return "disallow-me" not in url

    monkeypatch.setattr(cache, "get_parser", lambda url: FakeParser())
    assert cache.is_allowed("https://example.com/ok") is True
    assert cache.is_allowed("https://example.com/disallow-me") is False


def test_robots_unreadable_is_disallow(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = RobotsCache(user_agent="DataPilotAI-Test/0.1")
    monkeypatch.setattr(cache, "get_parser", lambda url: None)
    assert cache.is_allowed("https://example.com/page") is False


def test_collect_dry_run(tmp_path: Path) -> None:
    from src.ingestion.collector import DocumentCollector

    cfg = load_sources_config()
    collector = DocumentCollector(
        sources_cfg=cfg,
        output_dir=tmp_path / "raw",
        delay_seconds=0,
    )
    summary = collector.collect(limit=3, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["total_targets"] == 3
    assert summary["status_counts"].get("dry_run") == 3
    assert not (tmp_path / "raw" / "html").exists()

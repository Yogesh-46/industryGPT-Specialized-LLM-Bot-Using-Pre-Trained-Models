"""Flatten curated source inventory into collectible targets."""

from __future__ import annotations

from typing import Any, Iterator

from src.ingestion.models import TopicTarget


def iter_topic_targets(sources_cfg: dict[str, Any]) -> Iterator[TopicTarget]:
    """Yield ``TopicTarget`` entries from ``config/sources.yaml`` structure."""
    for source in sources_cfg.get("sources", []) or []:
        source_id = source["id"]
        source_name = source.get("name", source_id)
        category = source.get("category", "Unknown")
        license_note = source.get("license_note", "")
        for topic in source.get("topics", []) or []:
            yield TopicTarget(
                source_id=source_id,
                source_name=source_name,
                category=category,
                license_note=license_note,
                topic_id=topic["id"],
                title=topic.get("title", topic["id"]),
                url=topic["url"],
                topic=topic.get("topic", topic.get("title", topic["id"])),
            )


def list_topic_targets(sources_cfg: dict[str, Any]) -> list[TopicTarget]:
    """Return all curated topic targets as a list."""
    return list(iter_topic_targets(sources_cfg))

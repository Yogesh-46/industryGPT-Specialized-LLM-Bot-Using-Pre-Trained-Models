"""Shared dataclasses / schemas for ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TopicTarget:
    """A single curated documentation URL to collect."""

    source_id: str
    source_name: str
    category: str
    license_note: str
    topic_id: str
    title: str
    url: str
    topic: str

    @property
    def document_id(self) -> str:
        return f"{self.source_id}__{self.topic_id}"


@dataclass
class CollectionRecord:
    """Provenance + status for one collection attempt."""

    document_id: str
    source_id: str
    source_name: str
    category: str
    topic_id: str
    title: str
    topic: str
    url: str
    license_note: str
    collected_at: str
    status: str
    http_status: int | None = None
    content_type: str | None = None
    html_path: str | None = None
    meta_path: str | None = None
    content_length: int | None = None
    sha256: str | None = None
    robots_allowed: bool | None = None
    error: str | None = None
    skipped_reason: str | None = None
    inventory_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

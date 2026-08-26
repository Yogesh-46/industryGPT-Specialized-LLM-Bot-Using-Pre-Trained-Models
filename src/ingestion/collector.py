"""Document collection orchestration for Dataset A (raw corpus)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.ingestion.fetcher import DocumentFetcher, FetchError
from src.ingestion.inventory import list_topic_targets
from src.ingestion.models import CollectionRecord, TopicTarget, utc_now_iso
from src.ingestion.robots import RobotsCache
from src.utils.config import (
    get_request_delay_seconds,
    get_user_agent,
    load_sources_config,
)
from src.utils.paths import ensure_dir, project_root, resolve_path

logger = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel_or_abs(path: Path) -> str:
    """Return path relative to project root when possible, else absolute."""
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class DocumentCollector:
    """Collect curated documentation URLs into ``data/raw`` with provenance."""

    def __init__(
        self,
        sources_cfg: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        respect_robots: bool | None = None,
        delay_seconds: float | None = None,
        force: bool = False,
    ) -> None:
        self.sources_cfg = sources_cfg or load_sources_config()
        self.output_dir = resolve_path(output_dir or "./data/raw")
        self.html_dir = self.output_dir / "html"
        self.meta_dir = self.output_dir / "meta"
        self.manifest_path = self.output_dir / "collection_manifest.jsonl"
        self.summary_path = self.output_dir / "collection_summary.json"
        self.force = force

        politeness = self.sources_cfg.get("politeness", {})
        self.respect_robots = (
            politeness.get("respect_robots_txt", True)
            if respect_robots is None
            else respect_robots
        )
        self.user_agent = get_user_agent(self.sources_cfg)
        self.delay_seconds = (
            get_request_delay_seconds(self.sources_cfg)
            if delay_seconds is None
            else delay_seconds
        )
        self.max_retries = int(politeness.get("max_retries", 3))
        self.inventory_version = str(self.sources_cfg.get("inventory_version", "unknown"))

        self.robots = RobotsCache(self.user_agent)
        self.fetcher = DocumentFetcher(
            user_agent=self.user_agent,
            delay_seconds=self.delay_seconds,
            max_retries=self.max_retries,
        )

    def _paths_for(self, target: TopicTarget) -> tuple[Path, Path]:
        html_path = self.html_dir / target.source_id / f"{target.topic_id}.html"
        meta_path = self.meta_dir / target.source_id / f"{target.topic_id}.json"
        return html_path, meta_path

    def _already_collected(self, target: TopicTarget) -> bool:
        html_path, meta_path = self._paths_for(target)
        return html_path.exists() and meta_path.exists()

    def collect_one(self, target: TopicTarget) -> CollectionRecord:
        """Collect a single curated topic page."""
        html_path, meta_path = self._paths_for(target)
        collected_at = utc_now_iso()

        if not self.force and self._already_collected(target):
            record = CollectionRecord(
                document_id=target.document_id,
                source_id=target.source_id,
                source_name=target.source_name,
                category=target.category,
                topic_id=target.topic_id,
                title=target.title,
                topic=target.topic,
                url=target.url,
                license_note=target.license_note,
                collected_at=collected_at,
                status="skipped",
                html_path=_rel_or_abs(html_path),
                meta_path=_rel_or_abs(meta_path),
                skipped_reason="already_collected",
                inventory_version=self.inventory_version,
            )
            return record

        robots_allowed: bool | None = None
        if self.respect_robots:
            robots_allowed = self.robots.is_allowed(target.url)
            if not robots_allowed:
                record = CollectionRecord(
                    document_id=target.document_id,
                    source_id=target.source_id,
                    source_name=target.source_name,
                    category=target.category,
                    topic_id=target.topic_id,
                    title=target.title,
                    topic=target.topic,
                    url=target.url,
                    license_note=target.license_note,
                    collected_at=collected_at,
                    status="blocked_robots",
                    robots_allowed=False,
                    error="Disallowed by robots.txt or robots.txt unreadable",
                    inventory_version=self.inventory_version,
                )
                return record

        try:
            result = self.fetcher.fetch(target.url)
        except FetchError as exc:
            return CollectionRecord(
                document_id=target.document_id,
                source_id=target.source_id,
                source_name=target.source_name,
                category=target.category,
                topic_id=target.topic_id,
                title=target.title,
                topic=target.topic,
                url=target.url,
                license_note=target.license_note,
                collected_at=collected_at,
                status="error",
                robots_allowed=robots_allowed,
                error=str(exc),
                inventory_version=self.inventory_version,
            )

        if result.status_code != 200:
            return CollectionRecord(
                document_id=target.document_id,
                source_id=target.source_id,
                source_name=target.source_name,
                category=target.category,
                topic_id=target.topic_id,
                title=target.title,
                topic=target.topic,
                url=target.url,
                license_note=target.license_note,
                collected_at=collected_at,
                status="http_error",
                http_status=result.status_code,
                content_type=result.content_type,
                robots_allowed=robots_allowed,
                error=f"HTTP {result.status_code}",
                inventory_version=self.inventory_version,
                extra={"final_url": result.final_url},
            )

        ensure_dir(html_path.parent)
        html_path.write_bytes(result.content)

        digest = _sha256_bytes(result.content)
        record = CollectionRecord(
            document_id=target.document_id,
            source_id=target.source_id,
            source_name=target.source_name,
            category=target.category,
            topic_id=target.topic_id,
            title=target.title,
            topic=target.topic,
            url=target.url,
            license_note=target.license_note,
            collected_at=collected_at,
            status="collected",
            http_status=result.status_code,
            content_type=result.content_type,
            html_path=_rel_or_abs(html_path),
            meta_path=_rel_or_abs(meta_path),
            content_length=len(result.content),
            sha256=digest,
            robots_allowed=robots_allowed if robots_allowed is not None else True,
            inventory_version=self.inventory_version,
            extra={
                "final_url": result.final_url,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
        _write_json(meta_path, record.to_dict())
        return record

    def collect(
        self,
        *,
        limit: int | None = None,
        source_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Collect curated documents and write manifest + summary."""
        targets = list_topic_targets(self.sources_cfg)
        if source_id:
            targets = [t for t in targets if t.source_id == source_id]
        if limit is not None:
            targets = targets[: max(0, limit)]

        ensure_dir(self.output_dir)
        records: list[CollectionRecord] = []

        if dry_run:
            for target in targets:
                records.append(
                    CollectionRecord(
                        document_id=target.document_id,
                        source_id=target.source_id,
                        source_name=target.source_name,
                        category=target.category,
                        topic_id=target.topic_id,
                        title=target.title,
                        topic=target.topic,
                        url=target.url,
                        license_note=target.license_note,
                        collected_at=utc_now_iso(),
                        status="dry_run",
                        inventory_version=self.inventory_version,
                    )
                )
        else:
            # Each run writes a fresh manifest for the selected targets.
            # Restartability comes from skipping existing html+meta unless --force.
            if self.manifest_path.exists():
                self.manifest_path.unlink()

            for idx, target in enumerate(targets, start=1):
                logger.info(
                    "[%s/%s] %s — %s",
                    idx,
                    len(targets),
                    target.source_id,
                    target.url,
                )
                record = self.collect_one(target)
                records.append(record)
                _append_jsonl(self.manifest_path, record.to_dict())
                logger.info("  status=%s", record.status)

        summary = self._build_summary(records, dry_run=dry_run)
        if not dry_run:
            _write_json(self.summary_path, summary)
        return summary

    def _build_summary(
        self, records: list[CollectionRecord], *, dry_run: bool
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for record in records:
            counts[record.status] = counts.get(record.status, 0) + 1
            if record.status == "collected":
                by_source[record.source_id] = by_source.get(record.source_id, 0) + 1

        return {
            "inventory_version": self.inventory_version,
            "collected_at": utc_now_iso(),
            "dry_run": dry_run,
            "output_dir": _rel_or_abs(self.output_dir),
            "total_targets": len(records),
            "status_counts": counts,
            "collected_by_source": by_source,
            "respect_robots": self.respect_robots,
            "delay_seconds": self.delay_seconds,
            "user_agent": self.user_agent,
            "records": [r.to_dict() for r in records],
        }

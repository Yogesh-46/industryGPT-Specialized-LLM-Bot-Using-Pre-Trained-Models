"""Preprocessing pipeline: raw HTML → cleaned documents + stats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from src.preprocessing.cleaner import clean_html
from src.preprocessing.stats import (
    approximate_tokens,
    build_stats,
    dump_json,
    write_markdown_report,
    write_stats_figures,
)
from src.preprocessing.validators import DuplicateIndex, validate_document
from src.utils.config import load_yaml
from src.utils.paths import ensure_dir, project_root, resolve_path

logger = logging.getLogger(__name__)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_raw_records(raw_dir: Path) -> Iterator[tuple[dict[str, Any], Path]]:
    """Yield ``(metadata, html_path)`` pairs from ``data/raw/meta``."""
    meta_dir = raw_dir / "meta"
    html_dir = raw_dir / "html"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Raw metadata directory not found: {meta_dir}")

    for meta_path in sorted(meta_dir.rglob("*.json")):
        meta = _read_json(meta_path)
        rel = meta_path.relative_to(meta_dir)
        html_path = html_dir / rel.with_suffix(".html")
        if not html_path.exists():
            # Fallback to metadata-recorded path
            recorded = meta.get("html_path")
            if recorded:
                candidate = resolve_path(recorded)
                if candidate.exists():
                    html_path = candidate
        yield meta, html_path


class PreprocessPipeline:
    """Clean, validate, and export Dataset A documents."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_yaml("./config/preprocessing.yaml")
        paths = self.config.get("paths", {})
        self.raw_dir = resolve_path(paths.get("raw_dir", "./data/raw"))
        self.processed_dir = resolve_path(paths.get("processed_dir", "./data/processed"))
        self.kb_docs_dir = resolve_path(
            paths.get("knowledge_documents_dir", "./knowledge_base/documents")
        )
        self.stats_dir = resolve_path(paths.get("stats_dir", "./data/processed/stats"))

        out = self.config.get("output", {})
        self.documents_dir = self.processed_dir / out.get("documents_subdir", "documents")
        self.documents_jsonl = self.processed_dir / out.get(
            "documents_jsonl", "documents.jsonl"
        )
        self.rejected_jsonl = self.processed_dir / out.get(
            "rejected_jsonl", "rejected.jsonl"
        )
        self.stats_path = self.stats_dir / out.get(
            "stats_filename", "preprocessing_stats.json"
        )
        self.report_path = self.stats_dir / out.get(
            "report_filename", "preprocessing_report.md"
        )

    def run(self) -> dict[str, Any]:
        cleaning = self.config.get("cleaning", {})
        validation_cfg = self.config.get("validation", {})
        dup_cfg = self.config.get("duplicates", {})
        stats_cfg = self.config.get("statistics", {})
        chars_per_token = float(stats_cfg.get("approximate_tokens_chars_per_token", 4.0))

        ensure_dir(self.documents_dir)
        ensure_dir(self.kb_docs_dir)
        ensure_dir(self.stats_dir)

        # Fresh outputs each run
        for path in (self.documents_jsonl, self.rejected_jsonl):
            if path.exists():
                path.unlink()

        # Clear previous per-doc outputs
        for old in self.documents_dir.rglob("*.json"):
            old.unlink()
        for old in self.kb_docs_dir.rglob("*.json"):
            old.unlink()

        dup_index = DuplicateIndex(algorithm=dup_cfg.get("hash_algorithm", "sha256"))
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        records = list(iter_raw_records(self.raw_dir))
        logger.info("Found %s raw documents", len(records))

        for meta, html_path in records:
            document_id = meta.get("document_id") or (
                f"{meta.get('source_id')}__{meta.get('topic_id')}"
            )
            if not html_path.exists():
                item = {
                    "document_id": document_id,
                    "source_id": meta.get("source_id"),
                    "category": meta.get("category"),
                    "url": meta.get("url"),
                    "title": meta.get("title"),
                    "status": "rejected",
                    "rejection_reasons": [f"Missing HTML file: {html_path}"],
                    "validation_flags": {"malformed": True},
                    "is_duplicate": False,
                }
                rejected.append(item)
                _append_jsonl(self.rejected_jsonl, item)
                continue

            html = html_path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_html(
                html,
                remove_tags=list(cleaning.get("remove_tags", [])),
                main_content_selectors=list(
                    cleaning.get("main_content_selectors", [])
                ),
                collapse_whitespace=bool(cleaning.get("collapse_whitespace", True)),
                normalize_newlines=bool(cleaning.get("normalize_newlines", True)),
                strip_lines=bool(cleaning.get("strip_lines", True)),
                drop_empty_lines=bool(cleaning.get("drop_empty_lines", True)),
            )

            title = cleaned.get("title") or meta.get("title")
            text = cleaned.get("text") or ""
            metadata = {
                "document_id": document_id,
                "source_id": meta.get("source_id"),
                "source_name": meta.get("source_name"),
                "category": meta.get("category"),
                "topic_id": meta.get("topic_id"),
                "topic": meta.get("topic"),
                "title": title,
                "url": meta.get("url"),
                "license_note": meta.get("license_note"),
                "collected_at": meta.get("collected_at"),
                "inventory_version": meta.get("inventory_version"),
                "html_path": _rel(html_path),
            }

            validation = validate_document(
                text=text,
                metadata=metadata,
                parser_ok=bool(cleaned.get("parser_ok")),
                min_chars=int(validation_cfg.get("min_chars", 300)),
                max_chars=int(validation_cfg.get("max_chars", 500000)),
                min_words=int(validation_cfg.get("min_words", 50)),
                empty_after_clean_chars=int(
                    validation_cfg.get("empty_after_clean_chars", 50)
                ),
                require_title=bool(validation_cfg.get("require_title", True)),
                require_url=bool(validation_cfg.get("require_url", True)),
                require_source_id=bool(validation_cfg.get("require_source_id", True)),
            )

            is_dup, digest, original_id = False, None, None
            if dup_cfg.get("enabled", True) and text.strip():
                is_dup, digest, original_id = dup_index.check(document_id, text)

            char_count = len(text)
            word_count = len(text.split()) if text.strip() else 0
            token_count = approximate_tokens(text, chars_per_token)

            base_doc = {
                **metadata,
                "content": text,
                "char_count": char_count,
                "word_count": word_count,
                "approx_token_count": token_count,
                "content_hash": digest,
                "cleaner": {
                    "used_selector": cleaned.get("used_selector"),
                    "parser_ok": cleaned.get("parser_ok"),
                    "error": cleaned.get("error"),
                },
                "validation": validation.to_dict(),
                "is_duplicate": is_dup,
                "duplicate_of": original_id,
            }

            if (not validation.ok) or is_dup:
                reasons = list(validation.reasons)
                if is_dup:
                    reasons.append(f"Duplicate of {original_id}")
                item = {
                    **{k: v for k, v in base_doc.items() if k != "content"},
                    "status": "rejected",
                    "rejection_reasons": reasons,
                    "validation_flags": validation.flags,
                    # Keep a short preview for debugging, not full duplicate text dump necessity
                    "content_preview": text[:300],
                }
                rejected.append(item)
                _append_jsonl(self.rejected_jsonl, item)
                logger.info("Rejected %s: %s", document_id, "; ".join(reasons))
                continue

            accepted_doc = {**base_doc, "status": "accepted"}
            rel_name = Path(meta.get("source_id", "unknown")) / f"{meta.get('topic_id', document_id)}.json"
            processed_path = self.documents_dir / rel_name
            kb_path = self.kb_docs_dir / rel_name
            _write_json(processed_path, accepted_doc)
            _write_json(kb_path, accepted_doc)
            accepted_doc_out = {
                **accepted_doc,
                "processed_path": _rel(processed_path),
                "knowledge_path": _rel(kb_path),
            }
            accepted.append(accepted_doc_out)
            _append_jsonl(self.documents_jsonl, accepted_doc_out)
            logger.info(
                "Accepted %s (%s chars, selector=%s)",
                document_id,
                char_count,
                cleaned.get("used_selector"),
            )

        stats = build_stats(accepted, rejected, chars_per_token=chars_per_token)
        figure_paths: list[str] = []
        if stats_cfg.get("write_figures", True):
            figure_paths = write_stats_figures(stats, self.stats_dir / "figures")
            # Store relative paths when possible
            rel_figs = []
            for fp in figure_paths:
                p = Path(fp)
                try:
                    rel_figs.append(str(p.relative_to(project_root())))
                except ValueError:
                    rel_figs.append(fp)
            figure_paths = rel_figs

        stats["figures"] = figure_paths
        stats["outputs"] = {
            "documents_dir": _rel(self.documents_dir),
            "knowledge_documents_dir": _rel(self.kb_docs_dir),
            "documents_jsonl": _rel(self.documents_jsonl),
            "rejected_jsonl": _rel(self.rejected_jsonl),
            "stats_path": _rel(self.stats_path),
            "report_path": _rel(self.report_path),
        }
        dump_json(self.stats_path, stats)
        write_markdown_report(stats, self.report_path, figure_paths)
        return stats

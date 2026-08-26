"""Build chunk corpus from processed documents."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from src.preprocessing.chunking import chunk_document
from src.preprocessing.stats import dump_json
from src.utils.config import load_yaml
from src.utils.paths import ensure_dir, project_root, resolve_path

logger = logging.getLogger(__name__)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def _load_documents(processed_dir: Path) -> list[dict[str, Any]]:
    docs_dir = processed_dir / "documents"
    docs: list[dict[str, Any]] = []
    jsonl = processed_dir / "documents.jsonl"
    if jsonl.exists():
        with jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs
    for path in sorted(docs_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            docs.append(json.load(fh))
    return docs


class ChunkBuilder:
    """Create and persist RAG chunks from processed Dataset A documents."""

    def __init__(
        self,
        rag_config: dict[str, Any] | None = None,
        preprocessing_config: dict[str, Any] | None = None,
    ) -> None:
        self.rag_config = rag_config or load_yaml("./config/rag.yaml")
        self.pre_config = preprocessing_config or load_yaml(
            "./config/preprocessing.yaml"
        )
        paths = self.pre_config.get("paths", {})
        self.processed_dir = resolve_path(paths.get("processed_dir", "./data/processed"))
        self.chunks_dir = resolve_path("./knowledge_base/chunks")
        self.stats_dir = resolve_path(
            paths.get("stats_dir", "./data/processed/stats")
        )
        self.chunks_jsonl = self.chunks_dir / "chunks.jsonl"
        self.chunk_stats_path = self.stats_dir / "chunking_stats.json"

    def run(self) -> dict[str, Any]:
        chunk_cfg = self.rag_config.get("chunking", {})
        chars_per_token = float(
            self.pre_config.get("statistics", {}).get(
                "approximate_tokens_chars_per_token", 4.0
            )
        )

        ensure_dir(self.chunks_dir)
        ensure_dir(self.stats_dir)

        # Clear previous chunk outputs
        if self.chunks_jsonl.exists():
            self.chunks_jsonl.unlink()
        for old in self.chunks_dir.rglob("*.json"):
            if old.name != ".gitkeep":
                old.unlink()

        documents = _load_documents(self.processed_dir)
        logger.info("Chunking %s processed documents", len(documents))

        all_chunks: list[dict[str, Any]] = []
        by_category: Counter[str] = Counter()
        by_source: Counter[str] = Counter()

        with self.chunks_jsonl.open("w", encoding="utf-8") as out_fh:
            for doc in documents:
                chunks = chunk_document(
                    doc,
                    chunk_size_tokens=int(chunk_cfg.get("chunk_size_tokens", 650)),
                    chunk_overlap_tokens=int(chunk_cfg.get("chunk_overlap_tokens", 75)),
                    separators=list(chunk_cfg.get("separators", ["\n\n", "\n", ". ", " "])),
                    min_chunk_chars=int(chunk_cfg.get("min_chunk_chars", 200)),
                    max_chunk_chars=int(chunk_cfg.get("max_chunk_chars", 6000)),
                    chars_per_token=chars_per_token,
                )
                source_id = str(doc.get("source_id"))
                for chunk in chunks:
                    payload = chunk.to_dict()
                    all_chunks.append(payload)
                    by_category[chunk.category] += 1
                    by_source[chunk.source_id] += 1
                    out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

                    # Per-document mirror for inspection
                    doc_chunk_dir = self.chunks_dir / "by_document" / source_id
                    ensure_dir(doc_chunk_dir)
                    chunk_path = doc_chunk_dir / f"{chunk.chunk_id}.json"
                    with chunk_path.open("w", encoding="utf-8") as cfh:
                        json.dump(payload, cfh, indent=2, ensure_ascii=False)
                        cfh.write("\n")

                logger.info(
                    "%s -> %s chunks",
                    doc.get("document_id"),
                    len(chunks),
                )

        token_counts = [c["approx_token_count"] for c in all_chunks]
        char_counts = [c["char_count"] for c in all_chunks]
        stats = {
            "documents_chunked": len(documents),
            "final_chunk_count": len(all_chunks),
            "chunks_per_category": dict(by_category),
            "chunks_per_source": dict(by_source),
            "chunk_size_tokens": int(chunk_cfg.get("chunk_size_tokens", 650)),
            "chunk_overlap_tokens": int(chunk_cfg.get("chunk_overlap_tokens", 75)),
            "min_chunk_chars": int(chunk_cfg.get("min_chunk_chars", 200)),
            "max_chunk_chars": int(chunk_cfg.get("max_chunk_chars", 6000)),
            "separators": list(chunk_cfg.get("separators", [])),
            "strategy": chunk_cfg.get("strategy", "recursive_character"),
            "chars_per_token_assumption": chars_per_token,
            "approx_token_summary": {
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
                "mean": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
            },
            "char_summary": {
                "min": min(char_counts) if char_counts else 0,
                "max": max(char_counts) if char_counts else 0,
                "mean": (sum(char_counts) / len(char_counts)) if char_counts else 0.0,
            },
            "outputs": {
                "chunks_jsonl": _rel(self.chunks_jsonl),
                "chunks_dir": _rel(self.chunks_dir),
                "stats_path": _rel(self.chunk_stats_path),
            },
            "note": (
                "Chunk size/overlap are starting configuration values from "
                "config/rag.yaml and are not claimed to be optimal."
            ),
        }
        dump_json(self.chunk_stats_path, stats)

        # Update preprocessing stats placeholders when present
        pre_stats_path = self.stats_dir / "preprocessing_stats.json"
        if pre_stats_path.exists():
            with pre_stats_path.open("r", encoding="utf-8") as fh:
                pre_stats = json.load(fh)
            pre_stats["final_chunk_count"] = stats["final_chunk_count"]
            pre_stats["chunks_per_category"] = stats["chunks_per_category"]
            pre_stats["chunk_stats_status"] = "Computed"
            pre_stats["chunking_stats_path"] = _rel(self.chunk_stats_path)
            dump_json(pre_stats_path, pre_stats)

        return stats

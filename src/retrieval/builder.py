"""Build FAISS vector store from chunk corpus + embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.encoder import encoder_from_rag_config
from src.preprocessing.stats import dump_json
from src.retrieval.faiss_store import FaissVectorStore
from src.utils.config import load_yaml
from src.utils.paths import ensure_dir, project_root, resolve_path

logger = logging.getLogger(__name__)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def load_chunks(chunks_jsonl: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with chunks_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class VectorStoreBuilder:
    """Embed chunks and persist a rebuildable FAISS index."""

    def __init__(self, rag_config: dict[str, Any] | None = None) -> None:
        self.rag_config = rag_config or load_yaml("./config/rag.yaml")
        vs = self.rag_config.get("vector_store", {})
        self.persist_dir = resolve_path(
            vs.get("persist_dir", "./knowledge_base/vector_store")
        )
        self.index_filename = vs.get("index_filename", "faiss.index")
        self.metadata_filename = vs.get("metadata_filename", "chunks_metadata.jsonl")
        self.chunks_jsonl = resolve_path("./knowledge_base/chunks/chunks.jsonl")
        self.stats_path = resolve_path(
            "./data/processed/stats/vector_store_stats.json"
        )

    def run(self, *, show_progress: bool = True) -> dict[str, Any]:
        if not self.chunks_jsonl.exists():
            raise FileNotFoundError(
                f"Chunks not found: {self.chunks_jsonl}. Run scripts/build_chunks.py first."
            )

        chunks = load_chunks(self.chunks_jsonl)
        if not chunks:
            raise RuntimeError("No chunks available to index.")

        texts = [c.get("content") or "" for c in chunks]
        encoder = encoder_from_rag_config(self.rag_config)
        emb_cfg = self.rag_config.get("embeddings", {})
        retrieval_cfg = self.rag_config.get("retrieval", {})

        logger.info("Encoding %s chunks with %s", len(texts), encoder.model_id)
        # Prefer matrix encode without per-text disk cache for full rebuilds (faster on Windows).
        encoder.enable_disk_cache = False
        try:
            embeddings = encoder.encode(texts, show_progress=show_progress)
        except Exception:
            logger.exception("Embedding encode failed")
            raise

        # Metadata stored alongside index (include content for retrieval context)
        metadata = []
        for chunk in chunks:
            metadata.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "source": chunk.get("source"),
                    "source_id": chunk.get("source_id"),
                    "source_name": chunk.get("source_name"),
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "category": chunk.get("category"),
                    "topic": chunk.get("topic"),
                    "topic_id": chunk.get("topic_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "content": chunk.get("content"),
                    "char_count": chunk.get("char_count"),
                    "approx_token_count": chunk.get("approx_token_count"),
                    "license_note": chunk.get("license_note"),
                }
            )

        store = FaissVectorStore(
            self.persist_dir,
            index_filename=self.index_filename,
            metadata_filename=self.metadata_filename,
        )
        store.build(
            embeddings,
            metadata,
            model_id=encoder.model_id,
            index_type=str(retrieval_cfg.get("index_type", "IndexFlatIP")),
            normalize_already=bool(emb_cfg.get("normalize_embeddings", True)),
        )
        store.save()

        # Smoke retrieval with a domain query (does not claim evaluation quality)
        smoke_query = "What is a DISTKEY in Amazon Redshift?"
        query_vec = encoder.encode_query(smoke_query)
        hits = store.search(
            query_vec,
            top_k=int(retrieval_cfg.get("top_k", 5)),
        )
        smoke = [
            {
                "rank": h.rank,
                "score": h.score,
                "chunk_id": h.metadata.get("chunk_id"),
                "source_id": h.metadata.get("source_id"),
                "title": h.metadata.get("title"),
                "url": h.metadata.get("url"),
            }
            for h in hits
        ]

        stats = {
            "chunks_indexed": len(chunks),
            "embedding_dim": store.embedding_dim,
            "model_id": encoder.model_id,
            "normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", True)),
            "index_type": "IndexFlatIP",
            "top_k_default": int(retrieval_cfg.get("top_k", 5)),
            "selection_rationale": emb_cfg.get("selection_rationale"),
            "persist_dir": _rel(self.persist_dir),
            "index_path": _rel(store.index_path),
            "metadata_path": _rel(store.metadata_path),
            "config_path": _rel(store.config_path),
            "smoke_query": smoke_query,
            "smoke_hits": smoke,
            "note": (
                "Smoke retrieval verifies index wiring only; it is not an "
                "evaluation result."
            ),
        }
        ensure_dir(self.stats_path.parent)
        dump_json(self.stats_path, stats)
        logger.info("Vector store build complete: %s chunks", len(chunks))
        return stats

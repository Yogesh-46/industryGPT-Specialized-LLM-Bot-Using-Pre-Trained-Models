"""FAISS vector store with persistent index and metadata mapping."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    score: float
    rank: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class FaissVectorStore:
    """Persistent FAISS index with parallel metadata records.

    Designed for rebuildability from ``knowledge_base/chunks/chunks.jsonl``.
    Uses inner-product search; pair with L2-normalized embeddings for cosine.
    """

    def __init__(
        self,
        persist_dir: str | Path,
        *,
        index_filename: str = "faiss.index",
        metadata_filename: str = "chunks_metadata.jsonl",
        config_filename: str = "vector_store_config.json",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.index_path = self.persist_dir / index_filename
        self.metadata_path = self.persist_dir / metadata_filename
        self.config_path = self.persist_dir / config_filename
        self.index: Any | None = None
        self.metadata: list[dict[str, Any]] = []
        self.embedding_dim: int | None = None
        self.model_id: str | None = None

    def build(
        self,
        embeddings: np.ndarray,
        metadata: Sequence[dict[str, Any]],
        *,
        model_id: str,
        index_type: str = "IndexFlatIP",
        normalize_already: bool = True,
    ) -> None:
        """Build an in-memory index from embedding matrix + metadata rows."""
        import faiss

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (n, dim)")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("metadata length must match embedding rows")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if not normalize_already:
            faiss.normalize_L2(vectors)

        dim = int(vectors.shape[1])
        if index_type != "IndexFlatIP":
            logger.warning(
                "Configured index_type=%s; DataPilot V1 uses IndexFlatIP",
                index_type,
            )
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index = index
        self.metadata = [dict(m) for m in metadata]
        self.embedding_dim = dim
        self.model_id = model_id
        logger.info(
            "Built FAISS index: n=%s dim=%s model=%s",
            index.ntotal,
            dim,
            model_id,
        )

    def save(self) -> None:
        """Persist index, metadata JSONL, and config sidecar."""
        import faiss

        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as fh:
            for row in self.metadata:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        config = {
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "ntotal": int(self.index.ntotal),
            "index_type": "IndexFlatIP",
            "index_filename": self.index_path.name,
            "metadata_filename": self.metadata_path.name,
            "rebuildable": True,
        }
        with self.config_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
            fh.write("\n")
        logger.info("Saved vector store to %s", self.persist_dir)

    def load(self) -> None:
        """Load index + metadata from disk."""
        import faiss

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = []
        with self.metadata_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        if self.config_path.exists():
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.model_id = cfg.get("model_id")
            self.embedding_dim = cfg.get("embedding_dim")
        else:
            self.embedding_dim = int(self.index.d)

        if len(self.metadata) != int(self.index.ntotal):
            raise RuntimeError(
                f"Metadata/index size mismatch: {len(self.metadata)} vs {self.index.ntotal}"
            )
        logger.info(
            "Loaded FAISS index n=%s dim=%s",
            self.index.ntotal,
            self.embedding_dim,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievalHit]:
        """Search with a query embedding; return ranked hits with metadata."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        k = min(top_k, int(self.index.ntotal))
        if k <= 0:
            return []

        scores, indices = self.index.search(vector, k)
        hits: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            score_f = float(score)
            if score_threshold is not None and score_f < score_threshold:
                continue
            hits.append(
                RetrievalHit(
                    score=score_f,
                    rank=rank,
                    metadata=dict(self.metadata[int(idx)]),
                )
            )
        return hits

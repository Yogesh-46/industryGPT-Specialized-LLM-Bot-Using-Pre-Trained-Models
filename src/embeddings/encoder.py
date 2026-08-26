"""Configurable Sentence Transformers embedding encoder."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def resolve_device(device: str | None = "auto") -> str:
    """Resolve ``auto`` to ``cuda`` when available, else ``cpu``."""
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class EmbeddingEncoder:
    """Encode texts with a configurable Sentence Transformers model.

    Model id is read from config — never hard-coded at call sites.
    """

    def __init__(
        self,
        model_id: str,
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        enable_disk_cache: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_disk_cache = enable_disk_cache and self.cache_dir is not None
        self._model = model

        if self.enable_disk_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", self.model_id, self.device)
        self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    @property
    def embedding_dim(self) -> int | None:
        if self._model is None:
            return None
        try:
            return int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            return None

    def _cache_key(self, text: str) -> str:
        payload = f"{self.model_id}|{int(self.normalize_embeddings)}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, text: str) -> np.ndarray | None:
        if not self.enable_disk_cache or self.cache_dir is None:
            return None
        path = self.cache_dir / f"{self._cache_key(text)}.npy"
        if path.exists():
            return np.load(path)
        return None

    def _write_cache(self, text: str, vector: np.ndarray) -> None:
        if not self.enable_disk_cache or self.cache_dir is None:
            return
        path = self.cache_dir / f"{self._cache_key(text)}.npy"
        np.save(path, vector)

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to a float32 matrix of shape ``(n, dim)``."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._read_cache(text)
            if cached is not None:
                vectors[i] = cached.astype(np.float32, copy=False)
            else:
                missing_idx.append(i)
                missing_texts.append(text)

        if missing_texts:
            model = self._load_model()
            encoded = model.encode(
                missing_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            encoded = np.asarray(encoded, dtype=np.float32)
            for row, idx, text in zip(encoded, missing_idx, missing_texts):
                vectors[idx] = row
                self._write_cache(text, row)

        matrix = np.vstack([v for v in vectors if v is not None]).astype(np.float32)
        return matrix

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to shape ``(dim,)``."""
        matrix = self.encode([query], show_progress=False)
        return matrix[0]

    def selection_notes(self, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return model selection metadata for experiment logs."""
        cfg = cfg or {}
        return {
            "model_id": self.model_id,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "device": self.device,
            "selection_rationale": cfg.get("selection_rationale"),
            "alternative_candidates": cfg.get("alternative_candidates"),
        }


def encoder_from_rag_config(rag_cfg: dict[str, Any]) -> EmbeddingEncoder:
    """Construct an encoder from ``config/rag.yaml`` embeddings section."""
    emb = rag_cfg.get("embeddings", {})
    cache = rag_cfg.get("caching", {})
    return EmbeddingEncoder(
        model_id=str(emb.get("model_id")),
        normalize_embeddings=bool(emb.get("normalize_embeddings", True)),
        batch_size=int(emb.get("batch_size", 32)),
        device=str(emb.get("device", "auto")),
        cache_dir=cache.get("embeddings_cache_dir"),
        enable_disk_cache=bool(cache.get("enable_disk_cache", True)),
    )

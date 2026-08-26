"""Unit tests for embeddings + FAISS retrieval (mocked encoder; no GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings.encoder import EmbeddingEncoder
from src.retrieval.faiss_store import FaissVectorStore


class _FakeSentenceTransformer:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):  # noqa: ANN001
        # Deterministic pseudo-embeddings from text length / chars
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.normal(size=self.dim).astype(np.float32)
            if kwargs.get("normalize_embeddings", True):
                vec = vec / (np.linalg.norm(vec) + 1e-12)
            rows.append(vec)
        return np.vstack(rows)


def test_encoder_uses_injected_model_and_cache(tmp_path: Path) -> None:
    model = _FakeSentenceTransformer(dim=8)
    encoder = EmbeddingEncoder(
        model_id="fake/model",
        normalize_embeddings=True,
        batch_size=2,
        device="cpu",
        cache_dir=tmp_path / "cache",
        enable_disk_cache=True,
        model=model,
    )
    texts = ["DISTKEY sorts data", "SELECT joins tables"]
    first = encoder.encode(texts)
    assert first.shape == (2, 8)
    # Second call should hit cache (still works with same fake model)
    second = encoder.encode(texts)
    assert np.allclose(first, second)
    query = encoder.encode_query("DISTKEY")
    assert query.shape == (8,)


def test_faiss_store_build_save_load_search(tmp_path: Path) -> None:
    try:
        import faiss  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("faiss not installed")

    dim = 8
    n = 5
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    # normalize for IP ~ cosine
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    metadata = [
        {
            "chunk_id": f"c{i}",
            "document_id": f"d{i}",
            "source_id": "redshift",
            "title": f"Doc {i}",
            "url": f"https://example.com/{i}",
            "content": f"content {i} about DISTKEY and SORTKEY",
            "category": "Data Warehousing",
        }
        for i in range(n)
    ]

    store = FaissVectorStore(tmp_path / "vs")
    store.build(embeddings, metadata, model_id="fake/model")
    store.save()

    loaded = FaissVectorStore(tmp_path / "vs")
    loaded.load()
    assert loaded.index is not None
    assert loaded.index.ntotal == n

    hits = loaded.search(embeddings[0], top_k=3)
    assert len(hits) == 3
    assert hits[0].metadata["chunk_id"] == "c0"
    assert hits[0].score > hits[-1].score or len(hits) == 1
    assert "url" in hits[0].metadata

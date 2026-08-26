"""Unit tests for chunking (no GPU / no network)."""

from __future__ import annotations

from src.preprocessing.chunking import approx_token_count, chunk_document, recursive_split


def test_approx_token_count() -> None:
    assert approx_token_count("") == 0
    assert approx_token_count("abcd") == 1
    assert approx_token_count("a" * 40, chars_per_token=4.0) == 10


def test_recursive_split_respects_size_and_overlap() -> None:
    paragraphs = [f"Paragraph {i}. " + ("word " * 40) for i in range(12)]
    text = "\n\n".join(paragraphs)
    spans = recursive_split(
        text,
        chunk_size_tokens=50,
        chunk_overlap_tokens=10,
        separators=["\n\n", "\n", ". ", " "],
        chars_per_token=4.0,
    )
    assert len(spans) >= 2
    for chunk_text, start, end in spans:
        assert chunk_text
        assert end >= start
        # soft upper bound with some slack for overlap append behaviour
        assert len(chunk_text) <= int(50 * 4.0) + int(10 * 4.0) + 50


def test_chunk_document_preserves_metadata() -> None:
    doc = {
        "document_id": "postgresql__pg_select",
        "source_id": "postgresql",
        "source_name": "PostgreSQL Documentation",
        "category": "SQL",
        "topic_id": "pg_select",
        "topic": "SELECT",
        "title": "SELECT",
        "url": "https://www.postgresql.org/docs/current/sql-select.html",
        "license_note": "note",
        "content": ("The SELECT statement retrieves rows from tables.\n\n" * 80),
    }
    chunks = chunk_document(
        doc,
        chunk_size_tokens=80,
        chunk_overlap_tokens=10,
        min_chunk_chars=50,
        max_chunk_chars=6000,
        chars_per_token=4.0,
    )
    assert len(chunks) >= 2
    first = chunks[0]
    assert first.document_id == "postgresql__pg_select"
    assert first.source_id == "postgresql"
    assert first.category == "SQL"
    assert first.url.startswith("https://")
    assert first.chunk_id.endswith("__chunk_0000")
    assert first.content
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

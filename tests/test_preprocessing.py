"""Unit tests for preprocessing (no network / no GPU)."""

from __future__ import annotations

from pathlib import Path

from src.preprocessing.cleaner import clean_html
from src.preprocessing.stats import approximate_tokens, build_stats
from src.preprocessing.validators import DuplicateIndex, content_hash, validate_document


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
  <head><title>SELECT — Docs</title></head>
  <body>
    <nav>Home | Docs | Blog</nav>
    <main>
      <h1>SELECT</h1>
      <p>The SELECT statement retrieves rows from a table.</p>
      <p>You can use JOIN, WHERE, GROUP BY, and ORDER BY clauses.</p>
      <script>alert('ignore');</script>
      <style>.x{color:red}</style>
    </main>
    <footer>Copyright</footer>
  </body>
</html>
"""


def test_clean_html_removes_nav_script_and_keeps_main() -> None:
    result = clean_html(
        SAMPLE_HTML,
        remove_tags=["script", "style", "nav", "footer", "header"],
        main_content_selectors=["main", "article"],
    )
    assert result["parser_ok"] is True
    assert result["title"] == "SELECT — Docs"
    assert result["used_selector"] == "main"
    assert "SELECT statement retrieves rows" in result["text"]
    assert "alert(" not in result["text"]
    assert "Home | Docs | Blog" not in result["text"]
    assert "Copyright" not in result["text"]


def test_clean_html_normalizes_whitespace() -> None:
    html = "<html><body><p>Hello    world</p>\n\n\n<p>Next</p></body></html>"
    result = clean_html(html, remove_tags=["script"], main_content_selectors=["body"])
    assert "Hello world" in result["text"]
    assert "\n\n\n" not in result["text"]


def test_validate_document_flags_empty() -> None:
    result = validate_document(
        text="short",
        metadata={"title": "t", "url": "http://x", "source_id": "postgresql"},
        parser_ok=True,
        min_chars=300,
        max_chars=500000,
        min_words=50,
        empty_after_clean_chars=50,
        require_title=True,
        require_url=True,
        require_source_id=True,
    )
    assert result.ok is False
    assert result.flags["empty"] is True


def test_validate_document_accepts_good_text() -> None:
    text = " ".join(["token"] * 80)
    result = validate_document(
        text=text,
        metadata={"title": "Good", "url": "http://x", "source_id": "postgresql"},
        parser_ok=True,
        min_chars=300,
        max_chars=500000,
        min_words=50,
        empty_after_clean_chars=50,
        require_title=True,
        require_url=True,
        require_source_id=True,
    )
    # 80 tokens * 6 chars approx may be under min_chars; pad
    text = ("token " * 80) + ("x" * 200)
    result = validate_document(
        text=text,
        metadata={"title": "Good", "url": "http://x", "source_id": "postgresql"},
        parser_ok=True,
        min_chars=300,
        max_chars=500000,
        min_words=50,
        empty_after_clean_chars=50,
        require_title=True,
        require_url=True,
        require_source_id=True,
    )
    assert result.ok is True


def test_duplicate_index_detects_exact_duplicates() -> None:
    idx = DuplicateIndex()
    first = idx.check("doc_a", "Hello World")
    second = idx.check("doc_b", "hello   world")
    assert first[0] is False
    assert second[0] is True
    assert second[2] == "doc_a"
    assert content_hash("Hello World") == content_hash("hello   world")


def test_approximate_tokens_and_stats() -> None:
    assert approximate_tokens("abcd") == 1
    accepted = [
        {
            "source_id": "postgresql",
            "category": "SQL",
            "char_count": 100,
            "word_count": 20,
            "approx_token_count": 25,
        }
    ]
    rejected = [
        {
            "is_duplicate": True,
            "rejection_reasons": ["Duplicate of x"],
            "validation_flags": {"empty": False, "malformed": False},
        }
    ]
    stats = build_stats(accepted, rejected)
    assert stats["accepted_documents"] == 1
    assert stats["duplicate_count"] == 1
    assert stats["documents_per_source"]["postgresql"] == 1


def test_pipeline_on_tiny_raw_corpus(tmp_path: Path) -> None:
    from src.preprocessing.pipeline import PreprocessPipeline

    raw = tmp_path / "raw"
    html_dir = raw / "html" / "postgresql"
    meta_dir = raw / "meta" / "postgresql"
    html_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    html = SAMPLE_HTML + ("<p>More useful documentation content here.</p>" * 20)
    (html_dir / "pg_select.html").write_text(html, encoding="utf-8")
    meta = {
        "document_id": "postgresql__pg_select",
        "source_id": "postgresql",
        "source_name": "PostgreSQL Documentation",
        "category": "SQL",
        "topic_id": "pg_select",
        "topic": "SELECT",
        "title": "SELECT",
        "url": "https://www.postgresql.org/docs/current/sql-select.html",
        "license_note": "note",
        "collected_at": "2026-01-01T00:00:00+00:00",
        "inventory_version": "0.1.2",
    }
    import json

    (meta_dir / "pg_select.json").write_text(json.dumps(meta), encoding="utf-8")

    # Duplicate page
    (html_dir / "pg_select_dup.html").write_text(html, encoding="utf-8")
    meta_dup = {
        **meta,
        "document_id": "postgresql__pg_select_dup",
        "topic_id": "pg_select_dup",
    }
    (meta_dir / "pg_select_dup.json").write_text(json.dumps(meta_dup), encoding="utf-8")

    config = {
        "paths": {
            "raw_dir": str(raw),
            "processed_dir": str(tmp_path / "processed"),
            "knowledge_documents_dir": str(tmp_path / "kb"),
            "stats_dir": str(tmp_path / "processed" / "stats"),
        },
        "cleaning": {
            "remove_tags": ["script", "style", "nav", "footer", "header"],
            "main_content_selectors": ["main", "article"],
            "collapse_whitespace": True,
            "normalize_newlines": True,
            "strip_lines": True,
            "drop_empty_lines": True,
        },
        "validation": {
            "min_chars": 100,
            "max_chars": 500000,
            "min_words": 10,
            "empty_after_clean_chars": 20,
            "require_title": True,
            "require_url": True,
            "require_source_id": True,
        },
        "duplicates": {"enabled": True, "hash_algorithm": "sha256"},
        "statistics": {
            "approximate_tokens_chars_per_token": 4.0,
            "write_figures": False,
        },
        "output": {
            "documents_subdir": "documents",
            "documents_jsonl": "documents.jsonl",
            "rejected_jsonl": "rejected.jsonl",
            "stats_filename": "preprocessing_stats.json",
            "report_filename": "preprocessing_report.md",
        },
    }

    stats = PreprocessPipeline(config).run()
    assert stats["accepted_documents"] == 1
    assert stats["duplicate_count"] == 1
    assert (tmp_path / "processed" / "documents" / "postgresql" / "pg_select.json").exists()

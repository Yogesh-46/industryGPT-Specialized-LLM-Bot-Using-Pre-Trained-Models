"""Validation and duplicate-detection helpers for processed documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": self.reasons, "flags": self.flags}


def content_hash(text: str, algorithm: str = "sha256") -> str:
    """Hash normalized text for exact duplicate detection."""
    normalized = " ".join(text.split()).lower()
    h = hashlib.new(algorithm)
    h.update(normalized.encode("utf-8"))
    return h.hexdigest()


def validate_document(
    *,
    text: str,
    metadata: dict[str, Any],
    parser_ok: bool,
    min_chars: int,
    max_chars: int,
    min_words: int,
    empty_after_clean_chars: int,
    require_title: bool,
    require_url: bool,
    require_source_id: bool,
) -> ValidationResult:
    """Validate cleaned document content and required metadata fields."""
    reasons: list[str] = []
    flags = {
        "malformed": False,
        "empty": False,
        "too_short": False,
        "too_long": False,
        "missing_metadata": False,
    }

    if not parser_ok:
        flags["malformed"] = True
        reasons.append("HTML parse failed or malformed document")

    char_count = len(text)
    word_count = len(text.split()) if text.strip() else 0

    if char_count <= empty_after_clean_chars:
        flags["empty"] = True
        reasons.append(
            f"Empty/near-empty after cleaning ({char_count} chars <= {empty_after_clean_chars})"
        )

    if char_count < min_chars and not flags["empty"]:
        flags["too_short"] = True
        reasons.append(f"Below min_chars ({char_count} < {min_chars})")

    if word_count < min_words and not flags["empty"]:
        flags["too_short"] = True
        reasons.append(f"Below min_words ({word_count} < {min_words})")

    if char_count > max_chars:
        flags["too_long"] = True
        reasons.append(f"Above max_chars ({char_count} > {max_chars})")

    if require_title and not str(metadata.get("title") or "").strip():
        flags["missing_metadata"] = True
        reasons.append("Missing title")
    if require_url and not str(metadata.get("url") or "").strip():
        flags["missing_metadata"] = True
        reasons.append("Missing url")
    if require_source_id and not str(metadata.get("source_id") or "").strip():
        flags["missing_metadata"] = True
        reasons.append("Missing source_id")

    ok = not any(flags.values())
    return ValidationResult(ok=ok, reasons=reasons, flags=flags)


class DuplicateIndex:
    """Track exact-content duplicates within a preprocessing run."""

    def __init__(self, algorithm: str = "sha256") -> None:
        self.algorithm = algorithm
        self._seen: dict[str, str] = {}  # hash -> first document_id

    def check(self, document_id: str, text: str) -> tuple[bool, str, str | None]:
        """Return ``(is_duplicate, content_hash, original_document_id)``."""
        digest = content_hash(text, self.algorithm)
        if digest in self._seen:
            return True, digest, self._seen[digest]
        self._seen[digest] = document_id
        return False, digest, None

"""Configurable recursive text chunking with provenance metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


def approx_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Approximate token count without requiring a HF tokenizer."""
    if not text or not text.strip():
        return 0
    return max(1, int(round(len(text) / chars_per_token)))


@dataclass
class TextChunk:
    """A single chunk with provenance."""

    chunk_id: str
    document_id: str
    source: str
    source_id: str
    source_name: str
    title: str
    url: str
    category: str
    topic: str
    topic_id: str
    chunk_index: int
    content: str
    char_count: int
    approx_token_count: int
    char_start: int
    char_end: int
    license_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_keep_sep(text: str, separator: str) -> list[str]:
    if not separator:
        return list(text)
    if separator not in text:
        return [text]
    parts = text.split(separator)
    # Re-attach separator to all but last piece to preserve boundaries
    out: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            out.append(part + separator)
        elif part:
            out.append(part)
    return out


def recursive_split(
    text: str,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    separators: Sequence[str],
    chars_per_token: float = 4.0,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks.

    Returns list of ``(chunk_text, char_start, char_end)``.
    """
    if not text.strip():
        return []

    size_chars = max(1, int(chunk_size_tokens * chars_per_token))
    overlap_chars = max(0, int(chunk_overlap_tokens * chars_per_token))
    if overlap_chars >= size_chars:
        overlap_chars = max(0, size_chars // 5)

    pieces = _split_to_units(text, list(separators), size_chars)
    return _merge_units(pieces, text, size_chars=size_chars, overlap_chars=overlap_chars)


def _split_to_units(text: str, separators: list[str], size_chars: int) -> list[str]:
    """Recursively split oversized units using separator hierarchy."""
    if len(text) <= size_chars:
        return [text] if text else []

    if not separators:
        # Hard split by characters
        return [text[i : i + size_chars] for i in range(0, len(text), size_chars)]

    sep = separators[0]
    rest = separators[1:]
    parts = _split_keep_sep(text, sep) if sep else list(text)

    units: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= size_chars:
            units.append(part)
        else:
            units.extend(_split_to_units(part, rest, size_chars))
    return units


def _merge_units(
    units: list[str],
    original: str,
    *,
    size_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """Merge small units into chunks near ``size_chars`` with overlap."""
    if not units:
        return []

    chunks: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_len = 0
    cursor = 0  # approximate search position in original

    def flush(buffer: list[str]) -> None:
        nonlocal cursor
        if not buffer:
            return
        chunk_text = "".join(buffer).strip()
        if not chunk_text:
            return
        # Locate chunk in original for offsets (best-effort)
        start = original.find(chunk_text, max(0, cursor - len(chunk_text)))
        if start < 0:
            start = original.find(chunk_text)
        if start < 0:
            start = cursor
        end = start + len(chunk_text)
        chunks.append((chunk_text, start, end))
        cursor = max(cursor, start + 1)

    for unit in units:
        unit_len = len(unit)
        if current and current_len + unit_len > size_chars:
            flush(current)
            # Build overlap from end of previous chunk
            if overlap_chars > 0 and chunks:
                prev_text = chunks[-1][0]
                overlap = prev_text[-overlap_chars:]
                current = [overlap, unit]
                current_len = len(overlap) + unit_len
            else:
                current = [unit]
                current_len = unit_len
        else:
            current.append(unit)
            current_len += unit_len

    flush(current)
    return chunks


def chunk_document(
    document: dict[str, Any],
    *,
    chunk_size_tokens: int = 650,
    chunk_overlap_tokens: int = 75,
    separators: Sequence[str] | None = None,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 6000,
    chars_per_token: float = 4.0,
) -> list[TextChunk]:
    """Create provenance-preserving chunks for one processed document."""
    separators = list(
        separators
        if separators is not None
        else ["\n\n", "\n", ". ", " "]
    )
    content = document.get("content") or ""
    document_id = str(document.get("document_id"))
    spans = recursive_split(
        content,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        separators=separators,
        chars_per_token=chars_per_token,
    )

    chunks: list[TextChunk] = []
    for idx, (text, start, end) in enumerate(spans):
        if len(text) < min_chunk_chars and idx < len(spans) - 1:
            # Skip tiny non-final fragments when possible
            continue
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars]
            end = start + len(text)

        chunk = TextChunk(
            chunk_id=f"{document_id}__chunk_{idx:04d}",
            document_id=document_id,
            source=str(document.get("source_name") or document.get("source_id")),
            source_id=str(document.get("source_id")),
            source_name=str(document.get("source_name") or ""),
            title=str(document.get("title") or ""),
            url=str(document.get("url") or ""),
            category=str(document.get("category") or ""),
            topic=str(document.get("topic") or ""),
            topic_id=str(document.get("topic_id") or ""),
            chunk_index=idx,
            content=text,
            char_count=len(text),
            approx_token_count=approx_token_count(text, chars_per_token),
            char_start=start,
            char_end=end,
            license_note=document.get("license_note"),
        )
        chunks.append(chunk)

    # Re-index after possible skips
    for new_idx, chunk in enumerate(chunks):
        chunk.chunk_index = new_idx
        chunk.chunk_id = f"{document_id}__chunk_{new_idx:04d}"

    return chunks

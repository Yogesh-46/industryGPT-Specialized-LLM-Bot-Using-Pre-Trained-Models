"""Prompt construction for grounded RAG responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.utils.paths import resolve_path


def load_system_prompt(path: str | Path | None = None) -> str:
    prompt_path = resolve_path(path or "./config/prompts/system_prompt.txt")
    return prompt_path.read_text(encoding="utf-8").strip()


def format_context_blocks(
    hits: Sequence[dict[str, Any]],
    *,
    max_context_chars: int = 8000,
) -> str:
    """Format retrieved hits into a context block for the LLM."""
    if not hits:
        return "No retrieved context was available."

    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or hit
        title = meta.get("title") or "Untitled"
        source = meta.get("source_name") or meta.get("source") or meta.get("source_id")
        url = meta.get("url") or ""
        score = hit.get("score")
        content = (meta.get("content") or "").strip()
        header = f"[{i}] {title} | {source}"
        if score is not None:
            header += f" | score={float(score):.3f}"
        if url:
            header += f"\nURL: {url}"
        block = f"{header}\n{content}"
        if used + len(block) > max_context_chars and parts:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def build_user_prompt(query: str, context_block: str) -> str:
    return (
        "Use the retrieved documentation context below to answer the user question.\n"
        "Prefer the context for factual claims. If the context is insufficient, say so.\n"
        "Do not invent source URLs. Do not claim SQL was executed.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"User question:\n{query.strip()}\n"
    )


def build_chat_messages(
    *,
    system_prompt: str,
    query: str,
    hits: Sequence[dict[str, Any]],
    history: Sequence[dict[str, str]] | None = None,
    max_context_chars: int = 8000,
) -> list[dict[str, str]]:
    """Build chat messages for an instruction-tuned HF model."""
    context_block = format_context_blocks(hits, max_context_chars=max_context_chars)
    user_content = build_user_prompt(query, context_block)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(list(history))
    messages.append({"role": "user", "content": user_content})
    return messages


def format_sources(hits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact source list for UI / CLI display."""
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        meta = hit.get("metadata") or hit
        url = meta.get("url") or ""
        key = url or str(meta.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "rank": hit.get("rank"),
                "score": hit.get("score"),
                "title": meta.get("title"),
                "source_id": meta.get("source_id"),
                "source_name": meta.get("source_name") or meta.get("source"),
                "url": url,
                "chunk_id": meta.get("chunk_id"),
                "category": meta.get("category"),
                "topic": meta.get("topic"),
            }
        )
    return sources

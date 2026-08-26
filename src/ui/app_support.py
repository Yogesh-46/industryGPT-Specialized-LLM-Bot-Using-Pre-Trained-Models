"""Helpers for the Streamlit UI (no Streamlit import — unit-testable)."""

from __future__ import annotations

from typing import Any

from src.generation.prompts import format_sources
from src.generation.rag import BaselineRAGChatbot
from src.model.adapter import resolve_lora_adapter_path
from src.utils.config import load_yaml

EXAMPLE_QUESTIONS = [
    "Explain Slowly Changing Dimension Type 2.",
    "Why can a LEFT JOIN create duplicate rows?",
    "What is the difference between DISTKEY and SORTKEY?",
    "How should I design a retention dashboard?",
    "Write a Redshift-friendly pattern for monthly active users.",
]

MODE_LABELS = {
    "finetuned_rag": "Fine-tuned + RAG (System C)",
    "rag_only": "RAG only (System B)",
    "retrieve_only": "Retrieve only (no LLM)",
}


def default_mode() -> str:
    return "finetuned_rag" if resolve_lora_adapter_path() is not None else "rag_only"


def mode_settings(mode: str) -> dict[str, Any]:
    """Map UI mode to chatbot flags."""
    if mode == "finetuned_rag":
        adapter = resolve_lora_adapter_path()
        return {
            "use_rag": True,
            "adapter_path": str(adapter) if adapter is not None else None,
            "retrieve_only": False,
        }
    if mode == "retrieve_only":
        return {"use_rag": True, "adapter_path": None, "retrieve_only": True}
    return {"use_rag": True, "adapter_path": None, "retrieve_only": False}


def build_chatbot(mode: str, *, top_k: int | None = None) -> BaselineRAGChatbot:
    rag_cfg = load_yaml("./config/rag.yaml")
    model_cfg = load_yaml("./config/model.yaml")
    if top_k is not None:
        rag_cfg.setdefault("retrieval", {})["top_k"] = int(top_k)
    settings = mode_settings(mode)
    return BaselineRAGChatbot(
        rag_config=rag_cfg,
        model_config=model_cfg,
        adapter_path=settings["adapter_path"],
        use_rag=bool(settings["use_rag"]),
    )


def format_retrieve_only_answer(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return (
            "No relevant chunks were retrieved from the knowledge base. "
            "Try rephrasing the question."
        )
    lines = ["Retrieved documentation snippets (no LLM generation):", ""]
    for hit in hits:
        meta = hit.get("metadata") or {}
        title = meta.get("title") or "Untitled"
        content = " ".join(str(meta.get("content") or "").split())[:500]
        score = hit.get("score")
        score_s = f"{float(score):.3f}" if score is not None else "n/a"
        lines.append(f"**[{hit.get('rank')}] {title}** (score {score_s})")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def run_turn(
    bot: BaselineRAGChatbot,
    question: str,
    *,
    retrieve_only: bool,
) -> dict[str, Any]:
    """One user turn: retrieve-only or full ask()."""
    if retrieve_only:
        hits = bot.retrieve(question)
        return {
            "answer": format_retrieve_only_answer(hits),
            "sources": format_sources(hits),
            "hits": hits,
            "out_of_domain": False,
            "mode": "retrieve_only",
            "latency_ms": 0.0,
            "adapter_path": None,
        }
    return bot.ask(question)

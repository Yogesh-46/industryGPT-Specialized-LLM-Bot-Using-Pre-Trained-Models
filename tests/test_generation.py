"""Unit tests for generation / prompt / OOD (mocked LLM; no GPU)."""

from __future__ import annotations

from src.generation.domain import OOD_REFUSAL, is_out_of_domain
from src.generation.memory import ConversationMemory
from src.generation.prompts import (
    build_chat_messages,
    format_context_blocks,
    format_sources,
)


def test_ood_detects_medical_and_allows_bi_overlap() -> None:
    ood, cat = is_out_of_domain("How do I treat diabetes?")
    assert ood is True
    assert cat == "medical"

    ood2, _ = is_out_of_domain("What KPIs for a medical clinic dashboard?")
    assert ood2 is False


def test_prompt_includes_context_and_sources() -> None:
    hits = [
        {
            "rank": 1,
            "score": 0.71,
            "metadata": {
                "chunk_id": "c1",
                "title": "Distribution styles",
                "source_name": "Amazon Redshift Documentation",
                "source_id": "redshift",
                "url": "https://docs.aws.amazon.com/redshift/latest/dg/c_choosing_dist_sort.html",
                "content": "DISTKEY determines how rows are distributed.",
                "category": "Data Warehousing",
                "topic": "DISTKEY",
            },
        }
    ]
    messages = build_chat_messages(
        system_prompt="You are DataPilot AI.",
        query="What is DISTKEY?",
        hits=hits,
        history=[{"role": "user", "content": "Explain star schema."},
                 {"role": "assistant", "content": "A star schema has facts and dimensions."}],
        max_context_chars=2000,
    )
    assert messages[0]["role"] == "system"
    assert "DISTKEY" in messages[-1]["content"]
    assert "Retrieved context" in messages[-1]["content"]
    assert format_context_blocks(hits).startswith("[1]")
    sources = format_sources(hits)
    assert sources[0]["url"].startswith("https://")


def test_conversation_memory_bounds() -> None:
    mem = ConversationMemory(max_turns=2, max_history_chars=500)
    for i in range(6):
        mem.add("user", f"q{i}")
        mem.add("assistant", f"a{i}")
    assert len(mem.turns) <= 4


def test_rag_ask_with_mocked_generator(monkeypatch) -> None:  # noqa: ANN001
    from src.generation.rag import BaselineRAGChatbot

    bot = BaselineRAGChatbot(
        use_rag=True,
        generate_fn=lambda messages: "Mock grounded answer about DISTKEY.",
    )

    def fake_retrieve(query: str):
        return [
            {
                "rank": 1,
                "score": 0.6,
                "metadata": {
                    "chunk_id": "redshift__rs_diststyles__chunk_0000",
                    "title": "Distribution styles",
                    "source_id": "redshift",
                    "source_name": "Amazon Redshift Documentation",
                    "url": "https://example.com/dist",
                    "content": "DISTKEY distributes rows across nodes.",
                    "category": "Data Warehousing",
                    "topic": "DISTKEY",
                },
            }
        ]

    monkeypatch.setattr(bot, "retrieve", fake_retrieve)
    result = bot.ask("What is a DISTKEY?")
    assert "Mock grounded answer" in result["answer"]
    assert result["mode"] == "rag"
    assert result["sources"][0]["source_id"] == "redshift"
    assert result["out_of_domain"] is False

    ood = bot.ask("How do I diagnose my medical condition?")
    assert ood["out_of_domain"] is True
    assert OOD_REFUSAL in ood["answer"]

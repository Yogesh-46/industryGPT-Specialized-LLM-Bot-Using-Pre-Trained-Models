"""Tests for Streamlit helper logic (no browser / no LLM)."""

from __future__ import annotations

from src.model.adapter import resolve_lora_adapter_path
from src.ui.app_support import (
    EXAMPLE_QUESTIONS,
    default_mode,
    format_retrieve_only_answer,
    mode_settings,
)


def test_example_questions_are_in_domain() -> None:
    assert len(EXAMPLE_QUESTIONS) >= 4
    joined = " ".join(EXAMPLE_QUESTIONS).lower()
    assert "join" in joined or "distkey" in joined or "dashboard" in joined


def test_mode_settings_ft_uses_adapter_when_present() -> None:
    settings = mode_settings("finetuned_rag")
    assert settings["use_rag"] is True
    assert settings["retrieve_only"] is False
    adapter = resolve_lora_adapter_path()
    if adapter is not None:
        assert settings["adapter_path"] is not None
        assert default_mode() == "finetuned_rag"
    settings_b = mode_settings("rag_only")
    assert settings_b["adapter_path"] is None
    assert mode_settings("retrieve_only")["retrieve_only"] is True


def test_retrieve_only_formatter() -> None:
    empty = format_retrieve_only_answer([])
    assert "No relevant" in empty
    text = format_retrieve_only_answer(
        [
            {
                "rank": 1,
                "score": 0.42,
                "metadata": {"title": "DISTKEY", "content": "Distributes rows."},
            }
        ]
    )
    assert "DISTKEY" in text
    assert "no llm" in text.lower()

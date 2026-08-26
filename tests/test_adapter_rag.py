"""Tests for LoRA adapter resolution and System C wiring (no GPU)."""

from __future__ import annotations

from pathlib import Path

from src.generation.rag import BaselineRAGChatbot
from src.model.adapter import is_lora_adapter_dir, resolve_lora_adapter_path
from src.utils.config import load_yaml


def test_colab_adapter_is_detectable() -> None:
    cfg = load_yaml("./config/model.yaml")
    path = resolve_lora_adapter_path(cfg)
    assert path is not None
    assert path.name == "adapter"
    assert is_lora_adapter_dir(path)
    assert (path / "adapter_config.json").exists()
    assert (path / "adapter_model.safetensors").exists()
    configured = Path(str(cfg["paths"]["adapter_path"]))
    assert configured.as_posix().endswith("colab_t4_qlora_v1/adapter")


def test_finetuned_rag_mode_with_mock_generator(monkeypatch) -> None:  # noqa: ANN001
    adapter = resolve_lora_adapter_path()
    assert adapter is not None
    bot = BaselineRAGChatbot(
        use_rag=True,
        adapter_path=str(adapter),
        generate_fn=lambda messages: "FT+RAG mock answer.",
    )

    def fake_retrieve(query: str):
        return [
            {
                "rank": 1,
                "score": 0.5,
                "metadata": {
                    "chunk_id": "c1",
                    "title": "Star schema",
                    "source_id": "kimball",
                    "source_name": "Kimball",
                    "url": "https://example.com/star",
                    "content": "A star schema has a fact table and dimensions.",
                },
            }
        ]

    monkeypatch.setattr(bot, "retrieve", fake_retrieve)
    result = bot.ask("What is a star schema?")
    assert result["mode"] == "finetuned_rag"
    assert result["adapter_path"]
    assert "mock" in result["answer"].lower()

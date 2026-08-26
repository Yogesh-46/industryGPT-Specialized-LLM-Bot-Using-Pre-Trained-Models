"""Tests for QLoRA formatting and dry-run helpers (no GPU training)."""

from __future__ import annotations

from pathlib import Path

from src.finetuning.formatting import build_user_content, row_to_messages, row_to_text
from src.finetuning.train import dry_run_report, ensure_triton_ops_stub, load_model_config, resolve_use_4bit
from src.utils.config import load_yaml


def test_build_user_content_with_and_without_input() -> None:
    assert build_user_content("Explain JOINs.") == "Explain JOINs."
    assert "Input:" in build_user_content("Rewrite SQL.", "SELECT 1")


def test_row_to_messages_and_fallback_text() -> None:
    row = {
        "instruction": "What is a star schema?",
        "input": "",
        "response": "A star schema has a central fact table linked to dimensions.",
    }
    messages = row_to_messages(row)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    text = row_to_text(row, tokenizer=None)
    assert "<|im_start|>user" in text
    assert "star schema" in text.lower()
    assert "fact table" in text.lower()


def test_model_config_has_qlora_settings() -> None:
    cfg = load_yaml("./config/model.yaml")
    assert cfg["selected"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["quantization"]["load_in_4bit"] is True
    assert cfg["lora"]["r"] == 16
    assert cfg["training"]["num_epochs"] <= cfg["training"]["max_epochs_cap"]
    assert cfg["training"]["num_epochs"] != 25


def test_dry_run_against_dataset_b() -> None:
    cfg = load_model_config("./config/model.yaml")
    report = dry_run_report(
        model_cfg=cfg,
        train_path=Path("data/finetuning/train.jsonl"),
        val_path=Path("data/finetuning/validation.jsonl"),
    )
    assert report["ok"] is True
    assert report["train_examples"] >= 400
    assert report["validation_examples"] >= 20
    assert report["sample_formatted_preview"]


def test_no_4bit_flag_skips_quantization() -> None:
    cfg = load_model_config("./config/model.yaml")
    assert resolve_use_4bit(no_4bit=True, model_cfg=cfg) is False


def test_triton_ops_stub_is_importable() -> None:
    ensure_triton_ops_stub()
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

    assert callable(early_config_prune)
    assert callable(estimate_matmul_time)


def test_torchao_patch_does_not_raise() -> None:
    from src.finetuning.train import patch_incompatible_torchao

    patch_incompatible_torchao()

    class Dummy:
        def __init__(self, warmup_steps: float = 0, seed: int = 42) -> None:
            pass

    from src.finetuning.train import _supported_kwargs

    kept = _supported_kwargs(
        Dummy,
        {"warmup_ratio": 0.03, "warmup_steps": 0.03, "seed": 1, "tokenizer": "x"},
    )
    assert kept == {"warmup_steps": 0.03, "seed": 1}

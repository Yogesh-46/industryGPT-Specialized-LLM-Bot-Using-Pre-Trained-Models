"""Tests for locked V1 model selection config."""

from __future__ import annotations

from src.utils.config import load_yaml


def test_final_model_selection_is_locked() -> None:
    cfg = load_yaml("./config/model.yaml")
    selected = cfg["selected"]
    assert selected["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert selected["selection_status"] == "final_v1"
    assert selected["candidate_id"] == "qwen2.5-1.5b-instruct"
    decisions = {c["id"]: c.get("decision") for c in cfg["candidates"]}
    assert decisions["qwen2.5-1.5b-instruct"] == "selected"
    assert decisions["mistral-7b-instruct"] == "rejected_vram_risk"

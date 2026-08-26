"""Tests for evaluation dataset schema validation."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.schema import (
    TARGET_DISTRIBUTION,
    load_eval_jsonl,
    validate_eval_dataset,
    validate_eval_item,
)


def test_validate_item_catches_missing_fields() -> None:
    errors = validate_eval_item({"question_id": "x"}, index=0)
    assert errors
    assert any("missing field" in e for e in errors)


def test_eval_set_exists_and_matches_target_distribution() -> None:
    path = Path("data/evaluation/eval_set.jsonl")
    assert path.exists(), "Run scripts/prepare_evaluation_data.py first"
    items = load_eval_jsonl(path)
    report = validate_eval_dataset(items, enforce_distribution=True)
    assert report["ok"], report["errors"]
    assert report["total"] == 100
    assert report["by_category"] == TARGET_DISTRIBUTION

"""Tests for fine-tuning dataset isolation and validation."""

from __future__ import annotations

from pathlib import Path

from src.finetuning.dataset import (
    load_eval_question_set,
    load_jsonl,
    normalize_text,
    validate_ft_dataset,
)


def test_ft_dataset_exists_and_has_no_eval_leakage() -> None:
    path = Path("data/finetuning/all.jsonl")
    assert path.exists(), "Run scripts/prepare_finetuning_data.py first"
    rows = load_jsonl(path)
    assert len(rows) >= 500
    eval_qs = load_eval_question_set(Path("data/evaluation/eval_set.jsonl"))
    report = validate_ft_dataset(rows, eval_questions=eval_qs)
    assert report["ok"], report["errors"]
    assert report["leakage_count"] == 0
    # Direct check
    for row in rows:
        assert normalize_text(row["instruction"]) not in eval_qs


def test_train_val_files_exist() -> None:
    assert Path("data/finetuning/train.jsonl").exists()
    assert Path("data/finetuning/validation.jsonl").exists()
    train = load_jsonl(Path("data/finetuning/train.jsonl"))
    val = load_jsonl(Path("data/finetuning/validation.jsonl"))
    assert len(train) > len(val) > 0

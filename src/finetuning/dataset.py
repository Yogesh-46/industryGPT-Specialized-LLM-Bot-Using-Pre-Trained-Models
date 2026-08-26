"""Fine-tuning dataset validation and split helpers."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("instruction", "input", "response")


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_eval_question_set(eval_path: Path) -> set[str]:
    questions: set[str] = set()
    if not eval_path.exists():
        return questions
    for row in load_jsonl(eval_path):
        questions.add(normalize_text(str(row.get("question", ""))))
    return questions


def validate_ft_row(row: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"[{index}] missing field: {field}")
            continue
        if not isinstance(row[field], str):
            errors.append(f"[{index}] {field} must be string")
    if not str(row.get("instruction", "")).strip():
        errors.append(f"[{index}] empty instruction")
    if not str(row.get("response", "")).strip() or len(str(row.get("response", "")).strip()) < 20:
        errors.append(f"[{index}] response too short")
    if "input" in row and row["input"] is None:
        errors.append(f"[{index}] input must be string (use empty string)")
    return errors


def validate_ft_dataset(
    rows: list[dict[str, Any]],
    *,
    eval_questions: set[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    for i, row in enumerate(rows):
        errors.extend(validate_ft_row(row, index=i))

    keys = [
        (normalize_text(r.get("instruction", "")), normalize_text(r.get("input", "")))
        for r in rows
    ]
    if len(keys) != len(set(keys)):
        errors.append("duplicate instruction+input pairs detected")

    leaked: list[str] = []
    if eval_questions:
        for r in rows:
            inst = normalize_text(r.get("instruction", ""))
            if inst in eval_questions:
                leaked.append(inst)
        if leaked:
            errors.append(f"evaluation leakage: {len(leaked)} instructions match eval questions")

    categories = Counter(r.get("category", "unknown") for r in rows)
    return {
        "ok": not errors,
        "errors": errors,
        "total": len(rows),
        "by_category": dict(categories),
        "leakage_count": len(leaked),
    }


def train_val_split(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if shuffled else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val

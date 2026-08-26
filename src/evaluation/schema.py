"""Evaluation dataset schema helpers and validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "question_id",
    "category",
    "difficulty",
    "question",
    "expected_answer_points",
    "reference_answer",
    "source",
)

ALLOWED_CATEGORIES = {
    "SQL",
    "Data Engineering",
    "Data Warehousing",
    "BI/Dashboards",
    "Analytics",
    "Out-of-domain",
}

ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

TARGET_DISTRIBUTION = {
    "SQL": 25,
    "Data Engineering": 20,
    "Data Warehousing": 15,
    "BI/Dashboards": 15,
    "Analytics": 15,
    "Out-of-domain": 10,
}


def load_eval_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return items


def validate_eval_item(item: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"[{index}] missing field: {field}")

    if item.get("category") not in ALLOWED_CATEGORIES:
        errors.append(f"[{index}] invalid category: {item.get('category')}")
    if item.get("difficulty") not in ALLOWED_DIFFICULTIES:
        errors.append(f"[{index}] invalid difficulty: {item.get('difficulty')}")

    qid = item.get("question_id")
    if not isinstance(qid, str) or not qid.strip():
        errors.append(f"[{index}] question_id must be non-empty string")

    question = item.get("question")
    if not isinstance(question, str) or len(question.strip()) < 8:
        errors.append(f"[{index}] question too short")

    points = item.get("expected_answer_points")
    if not isinstance(points, list) or not points:
        errors.append(f"[{index}] expected_answer_points must be non-empty list")
    elif any(not isinstance(p, str) or not p.strip() for p in points):
        errors.append(f"[{index}] expected_answer_points must be non-empty strings")

    ref = item.get("reference_answer")
    if not isinstance(ref, str) or len(ref.strip()) < 20:
        errors.append(f"[{index}] reference_answer too short")

    source = item.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"[{index}] source must be non-empty string")

    return errors


def validate_eval_dataset(
    items: list[dict[str, Any]],
    *,
    enforce_distribution: bool = True,
) -> dict[str, Any]:
    """Validate schema, uniqueness, and optional category distribution."""
    errors: list[str] = []
    for i, item in enumerate(items):
        errors.extend(validate_eval_item(item, index=i))

    ids = [item.get("question_id") for item in items]
    if len(ids) != len(set(ids)):
        dupes = [qid for qid, c in Counter(ids).items() if c > 1]
        errors.append(f"duplicate question_id values: {dupes}")

    questions = [str(item.get("question", "")).strip().lower() for item in items]
    if len(questions) != len(set(questions)):
        errors.append("duplicate question text detected")

    by_category = Counter(item.get("category") for item in items)
    by_difficulty = Counter(item.get("difficulty") for item in items)

    if enforce_distribution:
        for cat, target in TARGET_DISTRIBUTION.items():
            actual = by_category.get(cat, 0)
            if actual != target:
                errors.append(
                    f"category {cat}: expected {target}, found {actual}"
                )

    return {
        "ok": not errors,
        "errors": errors,
        "total": len(items),
        "by_category": dict(by_category),
        "by_difficulty": dict(by_difficulty),
        "target_distribution": TARGET_DISTRIBUTION,
    }

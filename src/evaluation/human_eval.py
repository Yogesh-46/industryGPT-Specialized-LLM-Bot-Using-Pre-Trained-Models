"""Blank human-evaluation rubric (scores must be filled by annotators)."""

from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation.schema import load_eval_jsonl
from src.utils.paths import ensure_dir, resolve_path

RUBRIC_FIELDS = [
    "question_id",
    "category",
    "difficulty",
    "question",
    "system_id",
    "correctness_1to5",
    "relevance_1to5",
    "completeness_1to5",
    "groundedness_1to5",
    "notes",
]

SYSTEMS = ("baseline_llm", "rag_only", "finetuned_rag")


def write_human_eval_template(
    *,
    eval_path: str | Path = "./data/evaluation/eval_set.jsonl",
    output_path: str | Path = "./experiments/results/human_eval/rubric_blank.csv",
    systems: tuple[str, ...] = SYSTEMS,
) -> Path:
    """Write one blank row per (question, system). Leave score cells empty."""
    items = load_eval_jsonl(resolve_path(eval_path))
    out = resolve_path(output_path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUBRIC_FIELDS)
        writer.writeheader()
        for item in items:
            for system_id in systems:
                writer.writerow(
                    {
                        "question_id": item.get("question_id"),
                        "category": item.get("category"),
                        "difficulty": item.get("difficulty"),
                        "question": item.get("question"),
                        "system_id": system_id,
                        "correctness_1to5": "",
                        "relevance_1to5": "",
                        "completeness_1to5": "",
                        "groundedness_1to5": "",
                        "notes": "",
                    }
                )
    readme = out.parent / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Human evaluation (blank template)",
                "",
                "Score **only** after reading real system answers. Do not invent ratings.",
                "",
                "Scale: 1 = poor, 5 = excellent, for correctness, relevance, completeness, groundedness.",
                "",
                "Regenerate:",
                "",
                "```bash",
                "python scripts/prepare_human_eval.py",
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out

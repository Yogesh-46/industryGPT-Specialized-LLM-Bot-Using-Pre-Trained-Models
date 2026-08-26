#!/usr/bin/env python
"""Validate fine-tuning dataset and ensure no Dataset C leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetuning.dataset import (  # noqa: E402
    load_eval_question_set,
    load_jsonl,
    validate_ft_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./data/finetuning/all.jsonl")
    parser.add_argument(
        "--eval-path", default="./data/evaluation/eval_set.jsonl"
    )
    args = parser.parse_args()
    rows = load_jsonl(ROOT / args.path)
    eval_qs = load_eval_question_set(ROOT / args.eval_path)
    report = validate_ft_dataset(rows, eval_questions=eval_qs)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

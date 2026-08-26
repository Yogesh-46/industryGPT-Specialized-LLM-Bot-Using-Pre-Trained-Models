#!/usr/bin/env python
"""Validate held-out evaluation dataset schema and distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.schema import load_eval_jsonl, validate_eval_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./data/evaluation/eval_set.jsonl",
        help="Path to eval JSONL",
    )
    args = parser.parse_args()
    items = load_eval_jsonl(ROOT / args.path if not Path(args.path).is_absolute() else args.path)
    report = validate_eval_dataset(items, enforce_distribution=True)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

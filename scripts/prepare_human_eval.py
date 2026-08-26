#!/usr/bin/env python
"""Write a blank human-eval rubric CSV (no scores)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.human_eval import write_human_eval_template  # noqa: E402


def main() -> int:
    path = write_human_eval_template()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

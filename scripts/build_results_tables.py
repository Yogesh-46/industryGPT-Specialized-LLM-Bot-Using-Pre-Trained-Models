#!/usr/bin/env python
"""Write experiments/results/tables/results.md from executed artefacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.tables import write_results_tables  # noqa: E402


def main() -> int:
    path = write_results_tables()
    print(path)
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

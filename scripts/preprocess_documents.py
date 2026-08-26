#!/usr/bin/env python
"""Preprocess collected raw HTML into cleaned Dataset A documents.

Examples:
  python scripts/preprocess_documents.py
  python scripts/preprocess_documents.py --config config/preprocessing.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import PreprocessPipeline  # noqa: E402
from src.utils.config import load_env, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and validate raw documentation into data/processed/"
    )
    parser.add_argument(
        "--config",
        default="./config/preprocessing.yaml",
        help="Path to preprocessing.yaml",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_env()
    config = load_yaml(args.config)
    stats = PreprocessPipeline(config).run()
    print(json.dumps(stats, indent=2))

    if stats.get("accepted_documents", 0) == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

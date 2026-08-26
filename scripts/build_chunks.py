#!/usr/bin/env python
"""Build configurable text chunks from processed Dataset A documents.

Examples:
  python scripts/build_chunks.py
  python scripts/build_chunks.py --rag-config config/rag.yaml
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

from src.preprocessing.chunk_builder import ChunkBuilder  # noqa: E402
from src.utils.config import load_env, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk processed documents into knowledge_base/chunks/"
    )
    parser.add_argument("--rag-config", default="./config/rag.yaml")
    parser.add_argument("--preprocessing-config", default="./config/preprocessing.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_env()
    rag_cfg = load_yaml(args.rag_config)
    pre_cfg = load_yaml(args.preprocessing_config)
    stats = ChunkBuilder(rag_config=rag_cfg, preprocessing_config=pre_cfg).run()
    print(json.dumps(stats, indent=2))
    return 0 if stats.get("final_chunk_count", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

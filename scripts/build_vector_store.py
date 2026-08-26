#!/usr/bin/env python
"""Build persistent FAISS vector store from chunked Dataset A.

Examples:
  python scripts/build_vector_store.py
  python scripts/build_vector_store.py --rag-config config/rag.yaml
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

from src.retrieval.builder import VectorStoreBuilder  # noqa: E402
from src.utils.config import load_env, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks and build FAISS index under knowledge_base/vector_store/"
    )
    parser.add_argument("--rag-config", default="./config/rag.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable embedding progress bar",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_env()
    rag_cfg = load_yaml(args.rag_config)
    stats = VectorStoreBuilder(rag_config=rag_cfg).run(
        show_progress=not args.no_progress
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats.get("chunks_indexed", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Collect curated documentation pages listed in config/sources.yaml.

Examples:
  python scripts/collect_documents.py --dry-run
  python scripts/collect_documents.py --limit 2
  python scripts/collect_documents.py --source postgresql
  python scripts/collect_documents.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.collector import DocumentCollector  # noqa: E402
from src.utils.config import load_env, load_sources_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect curated BI/DE documentation into data/raw/"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to sources.yaml (default: config/sources.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/raw",
        help="Output directory for raw HTML + metadata",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Collect only one source id (e.g. postgresql, dbt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Collect at most N targets (after source filter)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if HTML + meta already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List targets without making network requests",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Do not check robots.txt (not recommended)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Override inter-request delay seconds",
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
    sources_cfg = load_sources_config(args.config)

    collector = DocumentCollector(
        sources_cfg=sources_cfg,
        output_dir=args.output_dir,
        respect_robots=not args.ignore_robots,
        delay_seconds=args.delay,
        force=args.force,
    )
    summary = collector.collect(
        limit=args.limit,
        source_id=args.source,
        dry_run=args.dry_run,
    )

    # Print compact summary (omit full records dump for readability)
    printable = {k: v for k, v in summary.items() if k != "records"}
    print(json.dumps(printable, indent=2))

    status_counts = summary.get("status_counts", {})
    failures = sum(
        status_counts.get(k, 0)
        for k in ("error", "http_error", "blocked_robots")
    )
    collected = status_counts.get("collected", 0) + status_counts.get("skipped", 0)
    if args.dry_run:
        return 0
    if collected == 0 and failures > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

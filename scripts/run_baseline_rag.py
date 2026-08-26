#!/usr/bin/env python
"""Run baseline RAG chatbot (Priority 1).

Examples:
  # Retrieve-only smoke test (no LLM download)
  python scripts/run_baseline_rag.py --retrieve-only "What is DISTKEY?"

  # One-shot RAG answer (loads base HF LLM)
  python scripts/run_baseline_rag.py "Explain star schema."

  # Interactive session
  python scripts/run_baseline_rag.py --interactive
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

from src.generation.rag import BaselineRAGChatbot  # noqa: E402
from src.model.adapter import resolve_lora_adapter_path  # noqa: E402
from src.utils.config import load_env, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataPilot AI baseline RAG CLI")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only run FAISS retrieval (no LLM generation)",
    )
    parser.add_argument("--no-rag", action="store_true", help="Base LLM only (System A)")
    parser.add_argument(
        "--finetuned-rag",
        action="store_true",
        help="System C: load LoRA adapter + RAG (default adapter from config)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Explicit adapter directory (implies --finetuned-rag)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def print_result(result: dict) -> None:
    print("\n=== Answer ===")
    print(result.get("answer", ""))
    sources = result.get("sources") or []
    if sources:
        print("\n=== Sources ===")
        for s in sources:
            score = s.get("score")
            score_s = f"{float(score):.3f}" if score is not None else "n/a"
            print(
                f"- [{s.get('rank')}] ({score_s}) {s.get('title')} | {s.get('url')}"
            )
    print(f"\nmode={result.get('mode')} latency_ms={result.get('latency_ms'):.1f}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_env()
    rag_cfg = load_yaml("./config/rag.yaml")
    model_cfg = load_yaml("./config/model.yaml")
    if args.top_k is not None:
        rag_cfg.setdefault("retrieval", {})["top_k"] = args.top_k

    adapter_path = None
    if args.adapter or args.finetuned_rag:
        resolved = resolve_lora_adapter_path(model_cfg, explicit=args.adapter)
        if resolved is None:
            print(
                "LoRA adapter not found. Set config/model.yaml paths.adapter_path "
                "or pass --adapter.",
                file=sys.stderr,
            )
            return 2
        adapter_path = str(resolved)
        logging.getLogger().info("Using adapter %s", adapter_path)

    bot = BaselineRAGChatbot(
        rag_config=rag_cfg,
        model_config=model_cfg,
        adapter_path=adapter_path,
        use_rag=not args.no_rag,
    )

    if args.retrieve_only:
        if not args.question:
            print("Provide a question with --retrieve-only", file=sys.stderr)
            return 2
        hits = bot.retrieve(args.question)
        print(json.dumps(hits, indent=2, ensure_ascii=False)[:8000])
        return 0

    if args.interactive:
        print("DataPilot AI (type 'exit' to quit, 'clear' to reset memory)")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            if q.lower() == "clear":
                bot.memory.clear()
                print("Memory cleared.")
                continue
            result = bot.ask(q)
            print_result(result)
        return 0

    if not args.question:
        parser_error = (
            "Provide a question, or use --interactive / --retrieve-only"
        )
        print(parser_error, file=sys.stderr)
        return 2

    result = bot.ask(args.question)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

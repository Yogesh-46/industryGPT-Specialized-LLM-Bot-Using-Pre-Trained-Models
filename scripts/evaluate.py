#!/usr/bin/env python
"""Run held-out evaluation for DataPilot AI systems A/B/C.

Examples:
  # Framework smoke test with mocked LLM (no GPU)
  python scripts/evaluate.py --systems rag_only --limit 5 --mock-llm

  # Real RAG evaluation subset (loads HF LLM)
  - Real System C eval (loads HF LLM; GPU recommended)
  python scripts/evaluate.py --systems finetuned_rag --limit 10

  # Baseline vs RAG (full set; expensive)
  python scripts/evaluate.py --systems baseline_llm,rag_only
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

from src.evaluation.runner import EvaluationRunner  # noqa: E402
from src.generation.domain import OOD_REFUSAL, is_out_of_domain  # noqa: E402
from src.utils.config import load_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataPilot AI evaluation runner")
    parser.add_argument(
        "--systems",
        default="rag_only",
        help="Comma-separated: baseline_llm,rag_only,finetuned_rag",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock generator (framework validation only)",
    )
    parser.add_argument(
        "--semantic-similarity",
        action="store_true",
        help="Compute embedding cosine similarity (loads sentence-transformers)",
    )
    parser.add_argument(
        "--experiment-id",
        default="eval_run",
        help="Experiment id prefix for output folder",
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments/results/evaluation",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter dir for finetuned_rag (default: config/model.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def mock_generate(messages: list[dict[str, str]]) -> str:
    """Deterministic mock answers for framework testing (not academic results)."""
    user = messages[-1]["content"] if messages else ""
    # Extract original question after marker when present
    q = user
    if "User question:" in user:
        q = user.split("User question:", 1)[1].strip()
    ood, _ = is_out_of_domain(q)
    if ood:
        return OOD_REFUSAL
    # Echo a short grounded-looking response using any retrieved context snippet
    snippet = ""
    if "Retrieved context:" in user:
        ctx = user.split("Retrieved context:", 1)[1]
        snippet = " ".join(ctx.split())[:220]
    if snippet and "No retrieved context" not in snippet:
        return (
            "Based on the retrieved documentation: "
            f"{snippet} "
            "I am not claiming this SQL was executed."
        )
    return (
        "This is a mock baseline answer for framework validation only. "
        "It is not an evaluated model response."
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_env()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    experiment_id = args.experiment_id
    if args.mock_llm and not experiment_id.startswith("mock_"):
        experiment_id = f"mock_{experiment_id}"

    runner = EvaluationRunner(
        output_dir=args.output_dir,
        adapter_path=args.adapter,
        generate_fn=mock_generate if args.mock_llm else None,
        compute_semantic_similarity=args.semantic_similarity,
    )
    summary = runner.run(
        systems=systems,
        limit=args.limit,
        experiment_id=experiment_id,
    )
    print(json.dumps(summary, indent=2))
    if args.mock_llm:
        print(
            "\nNOTE: --mock-llm results are for framework validation only "
            "and must not be reported as model performance.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

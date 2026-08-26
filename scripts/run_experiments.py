#!/usr/bin/env python
"""Run required GPU experiments exp_01 and/or exp_02.

These load the Hugging Face LLM and must not be confused with --mock-llm.

Examples (Colab T4 recommended):
  python scripts/run_experiments.py --exp exp_01 --limit 20
  python scripts/run_experiments.py --exp exp_01,exp_02
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
from src.evaluation.tables import write_results_tables  # noqa: E402
from src.utils.config import load_env  # noqa: E402

EXPERIMENTS = {
    "exp_01": ("exp_01_baseline_vs_rag", ["baseline_llm", "rag_only"]),
    "exp_02": ("exp_02_rag_vs_finetuned_rag", ["rag_only", "finetuned_rag"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataPilot AI GPU experiments")
    parser.add_argument(
        "--exp",
        default="exp_01,exp_02",
        help="Comma-separated: exp_01,exp_02",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="./experiments/results/evaluation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load Qwen in fp16 (Colab after uninstalling bitsandbytes)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    requested = [s.strip() for s in args.exp.split(",") if s.strip()]
    from src.utils.config import load_yaml

    model_cfg = load_yaml("./config/model.yaml")
    if args.no_4bit:
        model_cfg.setdefault("quantization", {})["load_in_4bit"] = False
        logging.info("Forcing fp16 load (load_in_4bit=false)")
    runner = EvaluationRunner(output_dir=args.output_dir, model_config=model_cfg)
    outputs: dict[str, object] = {}
    for key in requested:
        if key not in EXPERIMENTS:
            print(f"Unknown experiment: {key}", file=sys.stderr)
            return 2
        experiment_id, systems = EXPERIMENTS[key]
        logging.info("Starting %s systems=%s", experiment_id, systems)
        outputs[key] = runner.run(
            systems=systems,
            limit=args.limit,
            experiment_id=experiment_id,
        )
    tables = write_results_tables()
    print(json.dumps({"experiments": list(outputs), "tables": str(tables)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

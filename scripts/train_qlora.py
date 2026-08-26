#!/usr/bin/env python
"""Train DataPilot AI QLoRA adapter from Dataset B.

Preferred runtime: Google Colab T4 (see notebooks/04_qlora_finetune.ipynb).

Examples:
  # Validate data + formatting only (no GPU / no weight download)
  python scripts/train_qlora.py --dry-run

  # Colab T4 (current CUDA 12.8): skip 4-bit / bitsandbytes
  python scripts/train_qlora.py --run-name colab_t4_qlora_v1 --no-4bit

  # Smoke subset on GPU
  python scripts/train_qlora.py --max-train-samples 32 --max-val-samples 8 --run-name smoke_qlora
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

from src.finetuning.train import (  # noqa: E402
    dry_run_report,
    ensure_triton_ops_stub,
    load_model_config,
    run_qlora_training,
)
from src.utils.config import load_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataPilot AI QLoRA trainer")
    parser.add_argument("--config", default="./config/model.yaml")
    parser.add_argument("--train-path", default="./data/finetuning/train.jsonl")
    parser.add_argument("--val-path", default="./data/finetuning/validation.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base adapter directory (default from config/model.yaml)",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset + print format samples; do not train",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help=(
            "Train fp16 LoRA instead of 4-bit QLoRA. Required on current Colab "
            "(CUDA 12.8 / bitsandbytes CPU wheel + triton.ops crash)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    ensure_triton_ops_stub()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_cfg = load_model_config(args.config)
    if args.dry_run:
        report = dry_run_report(
            model_cfg=model_cfg,
            train_path=Path(args.train_path),
            val_path=Path(args.val_path),
        )
        print(json.dumps(report, indent=2))
        return 0

    summary = run_qlora_training(
        model_cfg=model_cfg,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        no_4bit=args.no_4bit,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "lora"}, indent=2))
    print(f"\nAdapter ready at: {summary.get('adapter_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

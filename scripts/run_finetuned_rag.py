#!/usr/bin/env python
"""Run System C: fine-tuned LoRA adapter + RAG.

Example:
  python scripts/run_finetuned_rag.py --retrieve-only "What is DISTKEY?"
  python scripts/run_finetuned_rag.py "Explain star schema."
  python scripts/run_finetuned_rag.py --interactive
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).with_name("run_baseline_rag.py")
    sys.argv = [str(script), "--finetuned-rag", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")

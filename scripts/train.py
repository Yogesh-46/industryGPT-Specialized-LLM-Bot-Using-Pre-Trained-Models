#!/usr/bin/env python
"""Alias for ``scripts/train_qlora.py`` (README compatibility)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_qlora.py")), run_name="__main__")

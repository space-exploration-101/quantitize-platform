#!/usr/bin/env python3
"""python -m quantitize.platform → 05_runner.py"""

from pathlib import Path
import runpy
import sys

_runner = Path(__file__).resolve().parent / "05_runner.py"
sys.argv[0] = str(_runner)
runpy.run_path(str(_runner), run_name="__main__")

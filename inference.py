#!/usr/bin/env python3
"""
Competition / CI entrypoint – run from repository root or inside Docker.

  python inference.py

Emits structured [START] / [STEP] / [END] blocks to stdout for the evaluator
(via ``openenv_farm.inference.run_baseline``). Summary JSON is printed after
episodes complete; use ``print(..., flush=True)`` throughout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Unbuffered output for CI/Docker capture of structured lines
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import after sys.path so ``openenv_farm`` resolves when running from /app
from openenv_farm.inference import run_baseline  # noqa: E402


def main() -> None:
    results: dict[str, Any] = {}
    try:
        results = run_baseline()
    except Exception as exc:
        print(f"[DEBUG] run_baseline() error: {exc}", flush=True)
    finally:
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

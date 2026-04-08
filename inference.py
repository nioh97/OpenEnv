#!/usr/bin/env python3
"""
Competition / CI entrypoint – run from repository root or inside Docker.

  python inference.py

Emits structured [START] / [STEP] / [END] blocks to stdout for the evaluator.
"""
from __future__ import annotations

import json
import os
import sys

# ── Force unbuffered stdout before ANY other work ────────────────────
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
except Exception:
    pass

# ── Required environment variables (OpenEnv Phase 2) ─────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Structured stdout helpers ────────────────────────────────────────
def _write(line: str) -> None:
    """Write a line to stdout fd 1 with fallback to raw fd write."""
    line_nl = line + "\n"
    try:
        sys.stdout.write(line_nl)
        sys.stdout.flush()
    except Exception:
        try:
            os.write(1, line_nl.encode())
        except Exception:
            pass


def log_start(task: str, env: str = "", model: str = "") -> None:
    _write(f"[START] task={task}")


def log_step(
    step: int,
    action: str = "",
    reward: float = 0.0,
    done: bool = False,
    error: Any = None,
) -> None:
    _write(f"[STEP] step={step} reward={reward}")


def log_end(
    task: str = "",
    success: bool = False,
    steps: int = 0,
    score: float = 0.0,
    rewards: list[float] | None = None,
) -> None:
    _write(f"[END] task={task} score={score} steps={steps}")


# ── Patch the package so run_baseline() uses OUR helpers ─────────────
import openenv_farm.inference as _farm_inf  # noqa: E402

_farm_inf.log_start = log_start  # type: ignore[assignment]
_farm_inf.log_step = log_step  # type: ignore[assignment]
_farm_inf.log_end = log_end  # type: ignore[assignment]

from openenv_farm.inference import run_baseline  # noqa: E402


def main() -> None:
    results: dict[str, Any] = {}
    try:
        results = run_baseline()
    except Exception as exc:
        _write(f"[DEBUG] run_baseline() error: {exc}")
    finally:
        _write(json.dumps(results, indent=2))
        try:
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()

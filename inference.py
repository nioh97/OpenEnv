"""
Competition / CI entrypoint: run from repository root.

  python inference.py

Requires repo root on ``sys.path`` so ``openenv_farm`` imports resolve.

Structured stdout (Phase 2) is defined here and bound into ``openenv_farm.inference``
so evaluators always see ``[START]`` / ``[STEP]`` / ``[END]`` on stdout from this file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Required environment variables (OpenEnv Phase 2) ─────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def log_start(task: str, env: str = "", model: str = "") -> None:
    _ = (env, model)  # sample API parity (BENCHMARK / MODEL_NAME)
    print(f"[START] task={task}", file=sys.stdout, flush=True)


def log_step(
    step: int,
    action: str = "",
    reward: float = 0.0,
    done: bool = False,
    error: Any = None,
) -> None:
    _ = (action, done, error)
    print(f"[STEP] step={step} reward={reward}", file=sys.stdout, flush=True)


def log_end(
    task: str = "",
    success: bool = False,
    steps: int = 0,
    score: float = 0.0,
    rewards: list[float] | None = None,
) -> None:
    _ = (success, rewards)
    print(f"[END] task={task} score={score} steps={steps}", file=sys.stdout, flush=True)


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import openenv_farm.inference as _farm_inference  # noqa: E402

_farm_inference.log_start = log_start
_farm_inference.log_step = log_step
_farm_inference.log_end = log_end

from openenv_farm.inference import run_baseline  # noqa: E402


def main() -> None:
    out = run_baseline()
    print(json.dumps(out, indent=2), file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()

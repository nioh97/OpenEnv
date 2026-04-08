"""
Competition / CI entrypoint: run from repository root.

  python inference.py

Requires repo root on ``sys.path`` so ``openenv_farm`` imports resolve.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Required environment variables (OpenEnv Phase 2) ─────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openenv_farm.inference import run_baseline  # noqa: E402


def main() -> None:
    out = run_baseline()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

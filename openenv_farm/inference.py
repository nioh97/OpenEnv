"""
Baseline agent: OpenAI chat with JSON actions + heuristic fallback.

Environment variables (OpenEnv Phase 2):
  ``API_BASE_URL``  – LLM endpoint      (default: https://api.openai.com/v1)
  ``MODEL_NAME``    – model identifier   (default: gpt-4o-mini)
  ``HF_TOKEN``      – Hugging Face token (no default)
  ``OPENAI_API_KEY`` – OpenAI key; used before ``HF_TOKEN`` when both are set.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from openenv_farm import config
from openenv_farm.env.farm_env import FarmEnv
from openenv_farm.env.models import Action, Observation
from openenv_farm.graders.profit_grader import grade as grade_profit
from openenv_farm.graders.sustainability_grader import grade as grade_sustainability
from openenv_farm.graders.yield_grader import grade as grade_yield
from openenv_farm.tasks import easy, hard, medium

# ── Required environment variables (OpenEnv Phase 2) ─────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("OPENAI_API_KEY", "") or HF_TOKEN or ""

BENCHMARK = "openenv_farm"
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_ORDER = (("easy", easy), ("medium", medium), ("hard", hard))


def _task_name_for(task_mod: Any) -> str:
    for n, m in TASK_ORDER:
        if m is task_mod:
            return n
    return "easy"


# ── Structured stdout helpers (OpenEnv Phase 2) ─────────────────────
def log_start(task: str, env: str = "", model: str = "") -> None:
    print(f"[START] task={task}", flush=True)


def log_step(step: int, action: str = "", reward: float = 0.0,
             done: bool = False, error: Any = None) -> None:
    print(f"[STEP] step={step} reward={reward}", flush=True)


def log_end(task: str = "", success: bool = False, steps: int = 0,
            score: float = 0.0, rewards: list[float] | None = None) -> None:
    print(f"[END] task={task} score={score} steps={steps}", flush=True)


SYSTEM_PROMPT = """You control a farm simulator. Each turn reply with ONLY a JSON object with keys:
irrigation (float liters >= 0),
fertilizer_n, fertilizer_p, fertilizer_k (float kg >= 0),
pesticide (boolean),
harvest (boolean).
If harvest is true, all other actions must be zero/false.
Choose actions to grow the crop and harvest when mature. Be concise."""


def _client() -> OpenAI | None:
    if not API_KEY:
        return None
    kwargs: dict[str, Any] = {"api_key": API_KEY}
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL
    return OpenAI(**kwargs)


def heuristic_action(env: FarmEnv, step_idx: int, obs: Observation) -> Action:
    _ = step_idx
    st = env.state()
    health = float(st.get("crop_health", 0.0))

    if obs.crop_stage == "mature" and health > 0.5:
        return Action(
            irrigation=0.0,
            fertilizer_n=0.0,
            fertilizer_p=0.0,
            fertilizer_k=0.0,
            pesticide=False,
            harvest=True,
        )

    water = float(obs.water_available)
    task = getattr(env, "_inference_task_name", "easy")
    liters_per_unit = {"easy": 100.0, "medium": 135.0, "hard": 150.0}.get(task, 100.0)
    if obs.soil_moisture < 0.45:
        irrigation = min(8.0 * liters_per_unit, water)
    elif obs.soil_moisture < 0.5:
        irrigation = min(4.0 * liters_per_unit, water)
    else:
        irrigation = 0.0

    day = int(obs.day)
    if day % 5 == 0:
        fertilizer_n = 1.5
        fertilizer_p = 1.0
        fertilizer_k = 1.0
    else:
        fertilizer_n = 0.0
        fertilizer_p = 0.0
        fertilizer_k = 0.0

    pesticide = bool(obs.pest_risk > 0.15)

    return Action(
        irrigation=float(irrigation),
        fertilizer_n=float(fertilizer_n),
        fertilizer_p=float(fertilizer_p),
        fertilizer_k=float(fertilizer_k),
        pesticide=pesticide,
        harvest=False,
    )


def _parse_action(raw: dict[str, Any] | None) -> Action | None:
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return Action.model_validate(raw)
    except Exception:
        return None


def _llm_action(client: OpenAI, model: str, obs: Observation) -> Action | None:
    user = json.dumps(obs.model_dump(), indent=2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(text)
        return _parse_action(data)
    except Exception:
        return None


def run_episode_for_task(
    task_mod: Any,
    *,
    seed: int = 42,
    max_steps: int | None = None,
    task_name: str | None = None,
) -> tuple[list[dict[str, Any]], str, list[float]]:
    """Returns (history, mode, rewards) where mode is 'openai' or 'heuristic'."""
    limit = max_steps if max_steps is not None else config.MAX_DAYS
    env = FarmEnv()
    conds = task_mod.get_initial_conditions()
    obs = env.reset(seed=seed, task_config=conds)
    tname = task_name if task_name is not None else _task_name_for(task_mod)
    setattr(env, "_inference_task_name", tname)

    client = _client()
    used_openai = False

    step_idx = 0
    rewards: list[float] = []
    while not env.done and step_idx < limit:
        action: Action | None = None
        if client is not None:
            action = _llm_action(client, MODEL_NAME, obs)
            if action is not None:
                used_openai = True
        if action is None:
            action = heuristic_action(env, step_idx, obs)
        obs, reward, done, _ = env.step(action)
        step_idx += 1
        rewards.append(reward.value)
        log_step(step=step_idx, reward=reward.value, done=done)

    mode = "openai" if used_openai else "heuristic"
    return env.history, mode, rewards


def aggregate_score(hist: list[dict[str, Any]], task_name: str) -> float:
    conds = {"easy": easy, "medium": medium, "hard": hard}[task_name].get_initial_conditions()
    y = grade_yield(hist, task_name=task_name)
    p = grade_profit(hist, task_name=task_name)
    s = grade_sustainability(
        hist,
        task_name=task_name,
        max_water=float(conds["water_available"]),
        max_fertilizer=max(1.0, float(conds["budget"]) / 6.0),
    )
    return (y + p + s) / 3.0


def run_baseline() -> dict[str, Any]:
    """
    Run baseline for all tasks. Scalar per task = mean(yield, profit, sustainability).

    Top-level keys: easy, medium, hard, plus optional ``meta`` for diagnostics.
    """
    out: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    base_seed = 42

    for i, (name, mod) in enumerate(TASK_ORDER):
        log_start(task=name, env=BENCHMARK, model=MODEL_NAME)
        hist, mode, rewards = run_episode_for_task(
            mod, seed=base_seed + i, task_name=name
        )
        final_score = float(aggregate_score(hist, name))
        final_score = min(max(final_score, 0.0), 1.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        out[name] = final_score
        meta[name] = {"steps": len(hist), "mode": mode}
        log_end(task=name, success=success, steps=len(hist),
                score=final_score, rewards=rewards)

    out["meta"] = meta
    return out


if __name__ == "__main__":
    out = run_baseline()
    print(json.dumps(out, indent=2))

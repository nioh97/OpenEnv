"""
Baseline agent: OpenAI chat with JSON actions + heuristic fallback.

Authentication (first non-empty wins): ``OPENAI_API_KEY``, then ``HF_TOKEN``
(for Hugging Face Inference / OpenAI-compatible endpoints).

Optional: ``API_BASE_URL`` (custom base URL), ``MODEL_NAME`` (default ``gpt-4o-mini``).
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

TASK_ORDER = (("easy", easy), ("medium", medium), ("hard", hard))


def _task_name_for(task_mod: Any) -> str:
    for n, m in TASK_ORDER:
        if m is task_mod:
            return n
    return "easy"


SYSTEM_PROMPT = """You control a farm simulator. Each turn reply with ONLY a JSON object with keys:
irrigation (float liters >= 0),
fertilizer_n, fertilizer_p, fertilizer_k (float kg >= 0),
pesticide (boolean),
harvest (boolean).
If harvest is true, all other actions must be zero/false.
Choose actions to grow the crop and harvest when mature. Be concise."""


def _client() -> OpenAI | None:
    key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )
    if not key:
        return None
    base = os.environ.get("API_BASE_URL", "").strip() or None
    kwargs: dict[str, Any] = {"api_key": key}
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


def _model_name() -> str:
    return os.environ.get("MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"


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
) -> tuple[list[dict[str, Any]], str]:
    """Returns (history, mode) where mode is 'openai' or 'heuristic'."""
    limit = max_steps if max_steps is not None else config.MAX_DAYS
    env = FarmEnv()
    conds = task_mod.get_initial_conditions()
    obs = env.reset(seed=seed, task_config=conds)
    tname = task_name if task_name is not None else _task_name_for(task_mod)
    setattr(env, "_inference_task_name", tname)

    client = _client()
    model = _model_name()
    used_openai = False

    step_idx = 0
    while not env.done and step_idx < limit:
        action: Action | None = None
        if client is not None:
            action = _llm_action(client, model, obs)
            if action is not None:
                used_openai = True
        if action is None:
            action = heuristic_action(env, step_idx, obs)
        obs, reward, _, _ = env.step(action)
        step_idx += 1
        print(f"[STEP] step={step_idx} reward={reward.value}", flush=True)

    mode = "openai" if used_openai else "heuristic"
    return env.history, mode


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
        print(f"[START] task={name}", flush=True)
        hist, mode = run_episode_for_task(
            mod, seed=base_seed + i, task_name=name
        )
        final_score = float(aggregate_score(hist, name))
        out[name] = final_score
        meta[name] = {"steps": len(hist), "mode": mode}
        print(
            f"[END] task={name} score={final_score} steps={len(hist)}",
            flush=True,
        )

    out["meta"] = meta
    return out


if __name__ == "__main__":
    out = run_baseline()
    print(json.dumps(out, indent=2))

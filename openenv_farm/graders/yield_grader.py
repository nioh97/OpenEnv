"""Yield-focused grader: harvest quality from terminal crop state."""

from __future__ import annotations

from typing import Any

STAGE_FACTOR: dict[str, float] = {
    "sowing": 0.2,
    "vegetative": 0.5,
    "flowering": 0.8,
    "mature": 1.0,
}

TASK_DIFFICULTY: dict[str, float] = {
    "easy": 1.0,
    "medium": 0.95,
    "hard": 0.9,
}

SCORE_FLOOR = 0.02


def _episode_state(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("state") or entry.get("state_after") or {}


def compute_yield_score(history: list[dict[str, Any]]) -> float:
    """
    Yield signal in [0, 1] before task-difficulty scaling.

    Partial credit when the episode did not end in harvest; full formula when harvested.
    """
    if not history:
        return 0.0

    last = history[-1]
    info = last.get("info") or {}
    action = last.get("action") or {}
    st = _episode_state(last)

    health = float(st.get("crop_health", 0.0))
    health = max(0.0, min(1.0, health))
    growth_progress = float(st.get("growth_progress", 0.0))
    growth_progress = max(0.0, min(1.0, growth_progress))
    stage = str(st.get("crop_stage", "sowing"))
    stage_factor = STAGE_FACTOR.get(stage, 0.2)

    harvested = info.get("termination_reason") == "harvest" and bool(action.get("harvest"))

    if not harvested:
        return float(min(health * stage_factor * 0.5, 0.5))

    yield_score = (0.85 * health + 0.15 * growth_progress) * stage_factor
    return float(min(yield_score, 1.0))


def grade(history: list[dict[str, Any]], task_name: str = "easy") -> float:
    """Deterministic score in [0, 1], scaled by task difficulty."""
    base = compute_yield_score(history)
    mult = TASK_DIFFICULTY.get(task_name, 1.0)
    score = max(SCORE_FLOOR, min(1.0, base * mult))
    return float(score)

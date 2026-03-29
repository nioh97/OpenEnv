"""Sustainability grader: yield quality plus input efficiency."""

from __future__ import annotations

from typing import Any

from openenv_farm.graders.yield_grader import TASK_DIFFICULTY, compute_yield_score

SCORE_FLOOR = 0.02

# Below both → treated as idle / hoarding (deterministic).
WATER_USAGE_THRESHOLD = 80.0
FERTILIZER_USAGE_THRESHOLD = 0.35


def _episode_state(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("state") or entry.get("state_after") or {}


def _totals(history: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """water_used, fertilizer_used (kg N+P+K), final_water, final_budget."""
    water_used = 0.0
    fertilizer_used = 0.0
    for entry in history:
        act = entry.get("action") or {}
        water_used += float(act.get("irrigation", 0.0))
        fertilizer_used += (
            float(act.get("fertilizer_n", 0.0))
            + float(act.get("fertilizer_p", 0.0))
            + float(act.get("fertilizer_k", 0.0))
        )
    last = history[-1]
    st = _episode_state(last)
    final_water = float(st.get("water_available", 0.0))
    final_budget = float(st.get("budget_remaining", 0.0))
    return water_used, fertilizer_used, final_water, final_budget


def _initial_water(water_used: float, final_water: float) -> float:
    return max(1.0, final_water + water_used)


def _initial_budget(history: list[dict[str, Any]], final_budget: float) -> float:
    spent = 0.0
    for entry in history:
        inf = entry.get("info") or {}
        spent += float(inf.get("step_water_charge", 0.0))
        spent += float(inf.get("step_fertilizer_charge", 0.0))
        spent += float(inf.get("step_pesticide_charge", 0.0))
    return max(1.0, final_budget + spent)


def grade(
    history: list[dict[str, Any]],
    *,
    task_name: str = "easy",
    max_water: float | None = None,
    max_fertilizer: float | None = None,
) -> float:
    """
    0.6 * yield + 0.25 * water_eff + 0.15 * fert_eff, minus idle penalty, task-scaled.

    Low yield short-circuits to discourage resource hoarding without production.
    SCORE_FLOOR applied only on the final return.
    """
    mult = TASK_DIFFICULTY.get(task_name, 1.0)
    raw: float

    if not history:
        raw = 0.0
    else:
        yield_score = compute_yield_score(history)
        if yield_score < 0.3:
            raw = min(1.0, yield_score * 0.8 * mult)
        else:
            water_used, fertilizer_used, final_water, final_budget = _totals(history)

            if water_used < WATER_USAGE_THRESHOLD and fertilizer_used < FERTILIZER_USAGE_THRESHOLD:
                penalty = 0.05
            else:
                penalty = 0.0

            mw = max_water if max_water is not None else _initial_water(water_used, final_water)
            mw = max(mw, 1.0)
            water_efficiency = 1.0 - (water_used / mw)
            water_efficiency = max(0.0, min(1.0, water_efficiency))

            mf = max_fertilizer
            if mf is None:
                ib = _initial_budget(history, final_budget)
                mf = max(1.0, ib / 8.0)

            mf = max(mf, 1.0)
            fertilizer_efficiency = 1.0 - (fertilizer_used / mf)
            fertilizer_efficiency = max(0.0, min(1.0, fertilizer_efficiency))

            score = (
                0.6 * yield_score
                + 0.25 * water_efficiency
                + 0.15 * fertilizer_efficiency
            )
            score -= penalty
            score = max(0.0, min(1.0, score))
            raw = score * mult

    return float(max(SCORE_FLOOR, min(1.0, raw)))

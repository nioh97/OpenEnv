"""Profit-oriented grader: revenue vs operational spend (deterministic)."""

from __future__ import annotations

from typing import Any

from openenv_farm import config
from openenv_farm.graders.yield_grader import TASK_DIFFICULTY

MAX_PROFIT = 1500.0
PROFIT_SCORE_CAP = 0.4
SCORE_FLOOR = 0.02
MIN_OPERATION_COST = 50.0
SHORT_EPISODE_SCORE = 0.05
EARLY_TERMINATION_DAY_THRESHOLD = 10


def _episode_state(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("state") or entry.get("state_after") or {}


def _avg_fertilizer_cost_per_kg() -> float:
    c = config.COSTS
    return (c["fertilizer_n_per_kg"] + c["fertilizer_p_per_kg"] + c["fertilizer_k_per_kg"]) / 3.0


def grade(history: list[dict[str, Any]], task_name: str = "easy") -> float:
    """Non-positive profit uses health fallback; positive profit scaled; exploit guards; capped; floor on final return."""
    if len(history) < 5:
        return float(max(SCORE_FLOOR, min(1.0, SHORT_EPISODE_SCORE)))

    last = history[-1]
    st = _episode_state(last)
    final_health = max(0.0, min(1.0, float(st.get("crop_health", 0.0))))
    revenue = final_health * 1000.0

    water_used = 0.0
    fertilizer_used = 0.0
    pesticide_cost = 0.0

    for entry in history:
        act = entry.get("action") or {}
        water_used += float(act.get("irrigation", 0.0))
        fertilizer_used += (
            float(act.get("fertilizer_n", 0.0))
            + float(act.get("fertilizer_p", 0.0))
            + float(act.get("fertilizer_k", 0.0))
        )
        inf = entry.get("info") or {}
        pesticide_cost += float(inf.get("step_pesticide_charge", 0.0))

    fert_unit = _avg_fertilizer_cost_per_kg()
    total_cost = (
        water_used * float(config.COSTS["water_per_liter"])
        + fertilizer_used * fert_unit
        + pesticide_cost
    )

    if total_cost < MIN_OPERATION_COST:
        total_cost = MIN_OPERATION_COST

    profit = revenue - total_cost
    if profit <= 0.0:
        base = 0.1 * final_health
    else:
        base = min(profit / MAX_PROFIT, 1.0)

    mult = TASK_DIFFICULTY.get(task_name, 1.0)
    score = base * mult

    final_day = int(st.get("current_day", st.get("day", 0)))
    if final_day < EARLY_TERMINATION_DAY_THRESHOLD:
        score *= 0.5

    score = min(score, PROFIT_SCORE_CAP)
    score = max(SCORE_FLOOR, min(1.0, score))
    return float(score)

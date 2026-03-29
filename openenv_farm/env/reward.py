"""Dense reward signal with an explicit breakdown for analysis and debugging."""

from __future__ import annotations

from typing import Any

from openenv_farm import config
from openenv_farm.env.models import Action, Reward


def _stage_rank(stage: str) -> int:
    order = ["sowing", "vegetative", "flowering", "mature"]
    return order.index(stage)


def compute_reward(
    prev: dict[str, Any],
    new: dict[str, Any],
    action: Action,
    *,
    step_water_charge: float,
    step_fertilizer_charge: float,
    step_pesticide_charge: float,
    terminated: bool,
    termination_reason: str | None,
) -> Reward:
    """Shaped reward from transition plus bookkeeping flags."""
    bd: dict[str, float] = {
        "yield_gain": 0.0,
        "health_improvement": 0.0,
        "stage_progression": 0.0,
        "growth_progress_gain": 0.0,
        "water_cost": 0.0,
        "fertilizer_cost": 0.0,
        "pesticide_cost": 0.0,
        "over_irrigation_penalty": 0.0,
        "over_fertilization_penalty": 0.0,
        "stress_penalty": 0.0,
        "pest_penalty": 0.0,
        "death_penalty": 0.0,
        "harvest_bonus": 0.0,
        "time_limit_penalty": 0.0,
        "alive_bonus": 0.0,
    }

    dh = float(new["crop_health"]) - float(prev["crop_health"])
    bd["health_improvement"] = 3.0 * dh
    stress = float(new.get("stress_index", 0.0))
    bd["stress_penalty"] = -config.PENALTY_STRESS * (stress**1.25) * 0.3

    pest_level = float(new.get("pest_level", 0.0))
    pest_penalty = -pest_level * 0.2
    bd["pest_penalty"] = pest_penalty

    dg = float(new["growth_progress"]) - float(prev["growth_progress"])
    bd["growth_progress_gain"] = 5.0 * max(0.0, dg)

    sr_prev = _stage_rank(str(prev["crop_stage"]))
    sr_new = _stage_rank(str(new["crop_stage"]))
    if sr_new > sr_prev:
        bd["stage_progression"] = 1.5 * float(sr_new - sr_prev)

    _cost_scale = 0.02
    bd["water_cost"] = -min(
        1.0,
        config.PENALTY_WATER_COST * max(0.0, step_water_charge) * _cost_scale,
    )
    bd["fertilizer_cost"] = -min(
        1.0,
        config.PENALTY_FERTILIZER_COST * max(0.0, step_fertilizer_charge) * _cost_scale,
    )
    if action.pesticide:
        bd["pesticide_cost"] = -min(
            1.0,
            config.PENALTY_PESTICIDE_COST * max(0.0, step_pesticide_charge) * _cost_scale,
        )

    if float(new["crop_health"]) > config.CROP_DEATH_THRESHOLD:
        bd["alive_bonus"] = 0.2

    mois = float(new["soil_moisture"])
    if mois > config.MOISTURE_OVERWATER_THRESHOLD:
        span = max(1e-6, 1.0 - config.MOISTURE_OVERWATER_THRESHOLD)
        bd["over_irrigation_penalty"] = -config.PENALTY_OVER_IRRIGATION * (
            (mois - config.MOISTURE_OVERWATER_THRESHOLD) / span
        ) ** 1.2

    npk = new["soil_nutrients"]
    for key in ("N", "P", "K"):
        excess = max(0.0, float(npk[key]) - config.NUTRIENT_TOXIC_THRESHOLD)
        if excess > 0:
            bd["over_fertilization_penalty"] += -config.PENALTY_OVER_FERTILIZATION * (excess**1.25)

    if termination_reason == "crop_death" or (
        terminated and float(new["crop_health"]) <= config.CROP_DEATH_THRESHOLD
    ):
        bd["death_penalty"] = -config.PENALTY_CROP_DEATH

    if action.harvest and termination_reason == "harvest":
        h = float(new["crop_health"])
        stage_mult = 0.25 + 0.25 * sr_new
        bd["harvest_bonus"] = (
            config.REWARD_HARVEST_BASE * stage_mult
            + config.REWARD_HARVEST_HEALTH_MULT * h
            + config.REWARD_HARVEST_STAGE_MULT * float(new["growth_progress"])
        )
        bd["yield_gain"] = bd["harvest_bonus"]

    if termination_reason == "max_days":
        bd["time_limit_penalty"] = -float(config.TIME_LIMIT_PENALTY)

    total = float(sum(bd.values()))
    return Reward(value=total, breakdown=bd)

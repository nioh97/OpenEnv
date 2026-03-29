"""Crop growth, stress, and stage progression (simple process-based model)."""

from __future__ import annotations

import math
from typing import Any

from openenv_farm import config
from openenv_farm.env import utils


def _stage_from_progress(progress: float) -> str:
    name = config.STAGE_THRESHOLDS[0][0]
    for stage_name, lo in config.STAGE_THRESHOLDS:
        if progress + 1e-9 >= lo:
            name = stage_name
    return name


def _moisture_factor(moisture: float) -> float:
    lo, hi = config.OPTIMAL_MOISTURE_RANGE
    center = 0.5 * (lo + hi)
    half_width = max(1e-6, 0.5 * (hi - lo))
    if moisture < lo:
        return utils.normalize(moisture, max(0.08, config.MOISTURE_STRESS_LOW - 0.04), lo)
    if moisture > hi:
        return utils.normalize(moisture, hi, config.MOISTURE_OVERWATER_THRESHOLD)
    d = abs(moisture - center) / half_width
    return float(math.cos(0.5 * math.pi * min(1.0, d))) ** 0.65


def _nutrient_factor(nutrients: dict[str, float]) -> float:
    n = nutrients["N"]
    p = nutrients["P"]
    k = nutrients["K"]
    lo, hi = config.TARGET_NUTRIENT_RANGE

    def contrib(x: float) -> float:
        if x < lo:
            return utils.normalize(x, 0.12, lo)
        if x > hi:
            return utils.normalize(x, hi, config.NUTRIENT_TOXIC_THRESHOLD)
        span = hi - lo
        mid = 0.5 * (lo + hi)
        return 1.0 - 0.20 * abs(2.0 * (x - mid) / span) ** 1.08

    return float((contrib(n) * contrib(p) * contrib(k)) ** (1.0 / 3.0))


def _pest_factor(pest_level: float) -> float:
    # Softer pest drag on growth (paired with reduced health damage below).
    return max(0.08, 1.0 - 0.46 * pest_level**1.25)


def _temperature_stress(temp_c: float) -> float:
    if temp_c < 8.0:
        return utils.normalize(temp_c, -2.0, 8.0)
    if temp_c > 30.0:
        return utils.normalize(temp_c, 30.0, 38.0)
    return 1.0


def update_crop_state(state: dict[str, Any], action: Any, weather: dict[str, float]) -> dict[str, Any]:
    """
    Advance crop health, growth progress, and stage from soil/pest context and weather.

    ``state`` must include soil_moisture, soil_nutrients (N,P,K), crop_health, pest_level,
    growth_progress, crop_stage.
    """
    _ = action
    m = float(state["soil_moisture"])
    nutrients = utils.dict_NPK(state["soil_nutrients"])
    health = float(state["crop_health"])
    pest = float(state["pest_level"])
    g = float(state["growth_progress"])

    f_m = _moisture_factor(m)
    f_n = _nutrient_factor(nutrients)
    f_p = _pest_factor(pest)
    f_t = _temperature_stress(float(weather["temperature"]))

    resource_index = f_m * f_n * f_p * f_t
    resource_index = float(utils.clamp(resource_index, 0.05, 1.12))

    carrying = max(1e-6, 1.0 - g)
    raw_delta = config.GROWTH_RATE_BASE * resource_index * carrying * (0.55 + 0.45 * health)

    stress = 1.0 - utils.clamp(f_m * f_n * f_p * f_t, 0.0, 1.0)
    stress = float(stress**1.05 * 0.72)
    health_damage = (
        config.HEALTH_DAMAGE_RATE * stress * 0.55
        + config.PEST_DAMAGE_TO_HEALTH * (pest**1.1) * 0.5
        + 0.009
        * max(0.0, m - config.MOISTURE_OVERWATER_THRESHOLD + 0.02)
        / max(1e-6, 1.0 - config.MOISTURE_OVERWATER_THRESHOLD)
    )
    health_recover = (
        config.HEALTH_RECOVERY_RATE * 1.25 * (resource_index**1.02) * (1.0 - 0.65 * pest)
    )

    health_new = utils.clamp(health - health_damage + health_recover, 0.0, 1.0)

    g_new = utils.clamp(g + raw_delta, 0.0, 0.9999)
    stage_new = _stage_from_progress(g_new)

    return {
        "crop_health": health_new,
        "growth_progress": g_new,
        "crop_stage": stage_new,
        "stress_index": float(stress),
        "resource_index": float(resource_index),
        "growth_delta": float(raw_delta),
    }

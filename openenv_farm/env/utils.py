"""Small numerical helpers and cost calculators."""

from __future__ import annotations

from typing import Mapping

from openenv_farm import config


def clamp(value: float, low: float, high: float) -> float:
    """Clamp scalar to [low, high]."""
    return max(low, min(high, value))


def normalize(value: float, low: float, high: float) -> float:
    """Linear map from [low, high] to [0, 1]; clamps to [0, 1] if outside."""
    if high <= low:
        return 0.5
    return clamp((value - low) / (high - low), 0.0, 1.0)


def water_cost(liters: float) -> float:
    return max(0.0, liters) * config.COSTS["water_per_liter"]


def fertilizer_cost(n_kg: float, p_kg: float, k_kg: float) -> float:
    return (
        max(0.0, n_kg) * config.COSTS["fertilizer_n_per_kg"]
        + max(0.0, p_kg) * config.COSTS["fertilizer_p_per_kg"]
        + max(0.0, k_kg) * config.COSTS["fertilizer_k_per_kg"]
    )


def pesticide_cost() -> float:
    return config.COSTS["pesticide_application"] + config.COSTS["pesticide_externalities"]


def dict_NPK(d: Mapping[str, float]) -> dict[str, float]:
    """Return a copy with N, P, K keys only, defaulting missing to 0."""
    return {
        "N": float(npk_get(d, "N")),
        "P": float(npk_get(d, "P")),
        "K": float(npk_get(d, "K")),
    }


def npk_get(d: Mapping[str, float], key: str) -> float:
    v = d.get(key, 0.0)
    return float(v)

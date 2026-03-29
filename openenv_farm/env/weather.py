"""Deterministic pseudo-random weather conditioned on seed and calendar day."""

from __future__ import annotations

import math
from typing import Any

from openenv_farm import config


def _mix64(x: int) -> int:
    """64-bit mixing function for deterministic pseudo-random integers."""
    x = (x ^ (x >> 33)) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 33)) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 33)) & 0xFFFFFFFFFFFFFFFF
    return x


def _u01(seed: int, day: int, salt: int) -> float:
    h = _mix64((seed * 0x9E3779B97F4A7C15 + day * 0xD6E8FEB866E53189 + salt) & 0xFFFFFFFFFFFFFFFF)
    return (h & 0xFFFFFFFF) / 4294967296.0


def generate_weather(seed: int, day: int) -> dict[str, float]:
    """
    Return today's weather for the given seed and day index.

    Temperature follows a smooth seasonal profile plus deterministic jitter.
    Rainfall is non-negative with heavy-tail-ish behavior (via log transform).
    """
    s = seed + config.WEATHER_SEED_OFFSET
    u1 = _u01(s, day, 1)
    u2 = _u01(s, day, 2)
    u3 = _u01(s, day, 3)

    seasonal = math.sin(2.0 * math.pi * (day / max(1, config.MAX_DAYS)))
    base_temp = 17.5 + 7.5 * seasonal
    temp = base_temp + (u1 - 0.5) * 5.5 + 0.8 * math.sin(day * 0.31)

    # Correlated rain: more likely in certain phases + bounded noise
    rain_score = -math.log(max(1e-12, u2)) * 2.6 - 1.85 + 0.35 * math.sin(day * 0.17 + u3 * 2.1)
    rainfall = max(0.0, min(32.0, rain_score * 4.2))

    return {"temperature": float(temp), "rainfall": float(rainfall)}


def get_forecast(seed: int, day: int, horizon: int = 3) -> list[dict[str, Any]]:
    """
    Next ``horizon`` days of forecast from the perspective of ``day``.

    Each item is ``{"temperature": float, "rainfall": float}``.
    The environment treats this as perfect knowledge of the generator (common in sims);
    agents may use it or ignore it.
    """
    out: list[dict[str, Any]] = []
    for h in range(1, horizon + 1):
        w = generate_weather(seed, day + h)
        out.append({"temperature": w["temperature"], "rainfall": w["rainfall"]})
    return out

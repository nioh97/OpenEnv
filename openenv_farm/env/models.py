"""Typed observation, action, and reward contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Observation(BaseModel):
    day: int = Field(ge=0)
    soil_moisture: float = Field(ge=0.0, le=1.0)
    soil_nutrients: dict[str, float]
    crop_stage: str
    pest_risk: float = Field(ge=0.0, le=1.0)
    water_available: float = Field(ge=0.0)
    budget_remaining: float
    weather_forecast: list[dict[str, Any]]

    @field_validator("soil_nutrients")
    @classmethod
    def validate_npk(cls, v: dict[str, float]) -> dict[str, float]:
        for key in ("N", "P", "K"):
            if key not in v:
                raise ValueError(f"soil_nutrients missing key '{key}'")
            x = float(v[key])
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"nutrient {key} must be in [0, 1], got {x}")
        if set(v.keys()) - {"N", "P", "K"}:
            raise ValueError("soil_nutrients must contain only N, P, K keys")
        return {k: float(v[k]) for k in ("N", "P", "K")}

    @field_validator("crop_stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        allowed = {"sowing", "vegetative", "flowering", "mature"}
        if v not in allowed:
            raise ValueError(f"crop_stage must be one of {allowed}")
        return v

    @field_validator("weather_forecast")
    @classmethod
    def validate_forecast(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(v) != 3:
            raise ValueError("weather_forecast must have length 3")
        out: list[dict[str, Any]] = []
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"forecast[{i}] must be a dict")
            if "temperature" not in item or "rainfall" not in item:
                raise ValueError(f"forecast[{i}] must include temperature and rainfall")
            out.append(
                {
                    "temperature": float(item["temperature"]),
                    "rainfall": float(max(0.0, float(item["rainfall"]))),
                }
            )
        return out


class Action(BaseModel):
    irrigation: float = Field(ge=0.0)
    fertilizer_n: float = Field(ge=0.0)
    fertilizer_p: float = Field(ge=0.0)
    fertilizer_k: float = Field(ge=0.0)
    pesticide: bool = False
    harvest: bool = False

    @model_validator(mode="after")
    def validate_harvest_exclusivity(self) -> Action:
        # Harvest is a terminal signal; discourage stacking heavy ops same day (soft contract)
        if self.harvest and (
            self.irrigation > 1e-6
            or self.fertilizer_n > 1e-6
            or self.fertilizer_p > 1e-6
            or self.fertilizer_k > 1e-6
            or self.pesticide
        ):
            raise ValueError("harvest must be the sole actionable decision for the day")
        return self


class Reward(BaseModel):
    value: float
    breakdown: dict[str, float]

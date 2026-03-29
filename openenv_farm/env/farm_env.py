"""Farm management MDP: soil, crop, pests, weather, and economics under constraints."""

from __future__ import annotations

import copy
from typing import Any

from openenv_farm import config
from openenv_farm.env import dynamics, reward, utils, weather
from openenv_farm.env.models import Action, Observation, Reward


class FarmEnv:
    """Production-style single-field farm simulator for RL agents."""

    def __init__(self) -> None:
        self._seed: int = 42
        self.current_day: int = 0
        self.soil_moisture: float = 0.0
        self.soil_nutrients: dict[str, float] = {"N": 0.0, "P": 0.0, "K": 0.0}
        self.crop_stage: str = "sowing"
        self.crop_health: float = 0.0
        self.pest_level: float = 0.0
        self.growth_progress: float = 0.0
        self.water_available: float = 0.0
        self.budget_remaining: float = 0.0
        self.done: bool = False
        self._termination_reason: str | None = None
        self.history: list[dict[str, Any]] = []
        self._last_stress_index: float = 0.0
        self._last_weather: dict[str, float] = {"temperature": 0.0, "rainfall": 0.0}

    def reset(self, seed: int = 42) -> Observation:
        self._seed = int(seed)
        self.current_day = 0
        self.soil_moisture = float(config.INITIAL_SOIL_MOISTURE)
        self.soil_nutrients = {k: float(v) for k, v in config.INITIAL_SOIL_NUTRIENTS.items()}
        self.crop_stage = str(config.INITIAL_CROP_STAGE)
        self.crop_health = float(config.INITIAL_CROP_HEALTH)
        self.pest_level = float(config.INITIAL_PEST_LEVEL)
        self.growth_progress = float(config.INITIAL_GROWTH_PROGRESS)
        self.water_available = float(config.INITIAL_WATER)
        self.budget_remaining = float(config.INITIAL_BUDGET)
        self.done = False
        self._termination_reason = None
        self.history = []
        self._last_stress_index = 0.0
        self._last_weather = {"temperature": 0.0, "rainfall": 0.0}
        return self._build_observation()

    def state(self) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "current_day": self.current_day,
            "soil_moisture": self.soil_moisture,
            "soil_nutrients": copy.deepcopy(self.soil_nutrients),
            "crop_stage": self.crop_stage,
            "crop_health": self.crop_health,
            "pest_level": self.pest_level,
            "growth_progress": self.growth_progress,
            "water_available": self.water_available,
            "budget_remaining": self.budget_remaining,
            "done": self.done,
            "termination_reason": self._termination_reason,
            "last_weather": copy.deepcopy(self._last_weather),
            "last_stress_index": self._last_stress_index,
        }

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode finished; call reset() before stepping.")

        if not isinstance(action, Action):
            action = Action.model_validate(action)

        info: dict[str, Any] = {
            "irrigation_applied": 0.0,
            "irrigation_requested": float(action.irrigation),
            "irrigation_clipped_water": False,
            "fertilizer_applied": {"N": 0.0, "P": 0.0, "K": 0.0},
            "fertilizer_requested": {
                "N": float(action.fertilizer_n),
                "P": float(action.fertilizer_p),
                "K": float(action.fertilizer_k),
            },
            "budget_blocked_fertilizer": False,
            "pesticide_blocked_budget": False,
            "pesticide_applied": bool(action.pesticide),
            "harvest": bool(action.harvest),
        }

        prev = self._snapshot_transition_prev()

        step_water_charge = 0.0
        step_fertilizer_charge = 0.0
        step_pesticide_charge = 0.0

        # 1) execute decisions (constraints as hard caps)
        applied_irrigation = float(action.irrigation)
        if applied_irrigation > self.water_available:
            info["irrigation_clipped_water"] = True
            applied_irrigation = float(self.water_available)
        info["irrigation_applied"] = applied_irrigation

        if applied_irrigation > 0.0:
            self.water_available = float(max(0.0, self.water_available - applied_irrigation))
            delta_m = (
                config.IRRIGATION_EFFICIENCY
                * applied_irrigation
                / max(1e-6, config.LITERS_TO_MOISTURE_SCALE)
            )
            self.soil_moisture = utils.clamp(self.soil_moisture + delta_m, 0.0, 1.0)
            step_water_charge = utils.water_cost(applied_irrigation)
            self.budget_remaining -= step_water_charge

        # Fertilizer: scale applications by affordable mass (proportional scale-down)
        n_req, p_req, k_req = action.fertilizer_n, action.fertilizer_p, action.fertilizer_k
        fert_cost_full = utils.fertilizer_cost(n_req, p_req, k_req)
        scale = 1.0
        if fert_cost_full > self.budget_remaining and fert_cost_full > 0.0:
            scale = max(0.0, float(self.budget_remaining) / fert_cost_full)
            info["budget_blocked_fertilizer"] = True

        n_ap = n_req * scale
        p_ap = p_req * scale
        k_ap = k_req * scale
        info["fertilizer_applied"] = {"N": n_ap, "P": p_ap, "K": k_ap}

        if n_ap > 0.0 or p_ap > 0.0 or k_ap > 0.0:
            self.soil_nutrients["N"] = utils.clamp(
                self.soil_nutrients["N"] + n_ap * config.FERTILIZER_YIELD_N,
                0.0,
                config.NUTRIENT_CAP,
            )
            self.soil_nutrients["P"] = utils.clamp(
                self.soil_nutrients["P"] + p_ap * config.FERTILIZER_YIELD_P,
                0.0,
                config.NUTRIENT_CAP,
            )
            self.soil_nutrients["K"] = utils.clamp(
                self.soil_nutrients["K"] + k_ap * config.FERTILIZER_YIELD_K,
                0.0,
                config.NUTRIENT_CAP,
            )
            step_fertilizer_charge = utils.fertilizer_cost(n_ap, p_ap, k_ap)
            self.budget_remaining -= step_fertilizer_charge

        if action.pesticide:
            charge = float(utils.pesticide_cost())
            if charge > self.budget_remaining:
                info["pesticide_blocked_budget"] = True
                action = action.model_copy(update={"pesticide": False})
                info["pesticide_applied"] = False
            else:
                self.pest_level = utils.clamp(
                    self.pest_level * (1.0 - config.PESTICIDE_REDUCTION), 0.0, 1.0
                )
                step_pesticide_charge = charge
                self.budget_remaining -= step_pesticide_charge

        # 5–6) weather + evaporation
        self._last_weather = weather.generate_weather(self._seed, self.current_day)
        rain = float(self._last_weather["rainfall"])
        temp = float(self._last_weather["temperature"])

        self.soil_moisture = utils.clamp(
            self.soil_moisture + rain * config.RAIN_TO_MOISTURE, 0.0, 1.0
        )

        decay = config.SOIL_MOISTURE_DECAY_BASE + config.TEMPERATURE_DECAY_FACTOR * max(
            0.0, temp - 22.0
        )
        self.soil_moisture = utils.clamp(self.soil_moisture - decay, 0.0, 1.0)

        # 7) crop dynamics
        dyn_state = {
            "soil_moisture": self.soil_moisture,
            "soil_nutrients": self.soil_nutrients,
            "crop_health": self.crop_health,
            "pest_level": self.pest_level,
            "growth_progress": self.growth_progress,
            "crop_stage": self.crop_stage,
        }
        crop_out = dynamics.update_crop_state(dyn_state, action, self._last_weather)
        self.crop_health = float(crop_out["crop_health"])
        self.growth_progress = float(crop_out["growth_progress"])
        self.crop_stage = str(crop_out["crop_stage"])
        self._last_stress_index = float(crop_out["stress_index"])
        info["resource_index"] = float(crop_out["resource_index"])
        info["growth_delta"] = float(crop_out["growth_delta"])

        # 8) pest dynamics (end-of-day pressure)
        pest_growth = config.PEST_NATURAL_GROWTH * (
            0.52 + config.PEST_MOISTURE_BOOST * self.soil_moisture
        ) + config.PEST_TEMP_BOOST * max(0.0, temp - 24.0)
        resilience = utils.clamp(0.45 + 0.55 * self.crop_health, 0.0, 1.0)
        self.pest_level = utils.clamp(
            self.pest_level + pest_growth * (1.0 - 0.55 * resilience), 0.0, 1.0
        )

        new = self._snapshot_transition_new()

        proposed_next_day = self.current_day + 1
        terminated = False
        termination_reason: str | None = None

        if action.harvest:
            terminated = True
            termination_reason = "harvest"
        elif self.crop_health < config.CROP_DEATH_THRESHOLD:
            terminated = True
            termination_reason = "crop_death"
        elif proposed_next_day >= config.MAX_DAYS:
            terminated = True
            termination_reason = "max_days"

        rew = reward.compute_reward(
            prev,
            new,
            action,
            step_water_charge=step_water_charge,
            step_fertilizer_charge=step_fertilizer_charge,
            step_pesticide_charge=step_pesticide_charge,
            terminated=terminated,
            termination_reason=termination_reason,
        )

        self.current_day = proposed_next_day
        self.done = terminated
        self._termination_reason = termination_reason

        obs = self._build_observation()

        info.update(
            {
                "day": obs.day,
                "weather": copy.deepcopy(self._last_weather),
                "termination_reason": termination_reason,
                "step_water_charge": step_water_charge,
                "step_fertilizer_charge": step_fertilizer_charge,
                "step_pesticide_charge": step_pesticide_charge,
            }
        )

        self.history.append(
            {
                "observation": obs.model_dump(),
                "action": action.model_dump(),
                "reward": rew.model_dump(),
                "info": copy.deepcopy(info),
                "state_after": self.state(),
            }
        )

        return obs, rew, self.done, info

    def _snapshot_transition_prev(self) -> dict[str, Any]:
        return {
            "crop_health": float(self.crop_health),
            "growth_progress": float(self.growth_progress),
            "crop_stage": str(self.crop_stage),
            "soil_moisture": float(self.soil_moisture),
            "soil_nutrients": utils.dict_NPK(self.soil_nutrients),
            "pest_level": float(self.pest_level),
            "stress_index": float(self._last_stress_index),
        }

    def _snapshot_transition_new(self) -> dict[str, Any]:
        return {
            "crop_health": float(self.crop_health),
            "growth_progress": float(self.growth_progress),
            "crop_stage": str(self.crop_stage),
            "soil_moisture": float(self.soil_moisture),
            "soil_nutrients": utils.dict_NPK(self.soil_nutrients),
            "pest_level": float(self.pest_level),
            "stress_index": float(self._last_stress_index),
        }

    def _build_observation(self) -> Observation:
        fc = weather.get_forecast(self._seed, self.current_day, horizon=3)
        return Observation(
            day=int(self.current_day),
            soil_moisture=float(utils.clamp(self.soil_moisture, 0.0, 1.0)),
            soil_nutrients=utils.dict_NPK(self.soil_nutrients),
            crop_stage=str(self.crop_stage),
            pest_risk=float(utils.clamp(self.pest_level, 0.0, 1.0)),
            water_available=float(max(0.0, self.water_available)),
            budget_remaining=float(self.budget_remaining),
            weather_forecast=fc,
        )

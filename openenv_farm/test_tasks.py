"""Smoke test: run each task with pseudo-random actions and print grader scores."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openenv_farm.env.farm_env import FarmEnv
from openenv_farm.env.models import Action
from openenv_farm.graders.profit_grader import grade as grade_profit
from openenv_farm.graders.sustainability_grader import grade as grade_sustainability
from openenv_farm.graders.yield_grader import grade as grade_yield
from openenv_farm.tasks import easy, hard, medium


def apply_initial_conditions(env: FarmEnv, conds: dict[str, Any]) -> None:
    env.soil_moisture = float(conds["soil_moisture"])
    env.soil_nutrients = {
        "N": float(conds["soil_nutrients"]["N"]),
        "P": float(conds["soil_nutrients"]["P"]),
        "K": float(conds["soil_nutrients"]["K"]),
    }
    env.pest_level = float(conds["pest_level"])
    env.water_available = float(conds["water_available"])
    env.budget_remaining = float(conds["budget"])


def sample_action(rng: random.Random, env: FarmEnv) -> Action:
    if rng.random() < 0.06:
        return Action(
            irrigation=0.0,
            fertilizer_n=0.0,
            fertilizer_p=0.0,
            fertilizer_k=0.0,
            pesticide=False,
            harvest=True,
        )
    irr = rng.choice([0.0, 120.0, 280.0, 450.0])
    irr = min(irr, float(env.water_available))
    fn = rng.choice([0.0, 0.0, 2.0, 5.0])
    fp = rng.choice([0.0, 0.0, 2.0, 4.0])
    fk = rng.choice([0.0, 0.0, 2.0, 4.0])
    pest = rng.random() < 0.12
    return Action(
        irrigation=float(irr),
        fertilizer_n=float(fn),
        fertilizer_p=float(fp),
        fertilizer_k=float(fk),
        pesticide=bool(pest),
        harvest=False,
    )


def run_episode(task_mod: Any, seed: int, max_steps: int = 40) -> list[dict[str, Any]]:
    env = FarmEnv()
    env.reset(seed=seed)
    apply_initial_conditions(env, task_mod.get_initial_conditions())
    rng = random.Random(seed + 17)

    for _ in range(max_steps):
        if env.done:
            break
        act = sample_action(rng, env)
        _, _, done, _ = env.step(act)
        if done:
            break

    return env.history


def main() -> None:
    tasks = [("easy", easy), ("medium", medium), ("hard", hard)]
    for name, mod in tasks:
        hist = run_episode(mod, seed=42 if name == "easy" else 43 if name == "medium" else 44)
        y = grade_yield(hist, task_name=name)
        p = grade_profit(hist, task_name=name)
        conds = mod.get_initial_conditions()
        s = grade_sustainability(
            hist,
            task_name=name,
            max_water=float(conds["water_available"]),
            max_fertilizer=max(1.0, float(conds["budget"]) / 6.0),
        )
        print(f"=== task={name} | steps={len(hist)} ===")
        print(f"  yield_grader:          {y:.4f}")
        print(f"  profit_grader:         {p:.4f}")
        print(f"  sustainability_grader: {s:.4f}")


if __name__ == "__main__":
    main()

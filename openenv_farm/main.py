"""Quick local smoke test for the farm environment (not part of the RL API surface)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openenv_farm.env.farm_env import FarmEnv
from openenv_farm.env.models import Action


def _rule_based_policy(day: int, obs) -> Action:
    # Simple heuristic: irrigate when dry, fertilize when low NPK, pesticide if pests spike.
    m = float(obs.soil_moisture)
    n = float(obs.soil_nutrients["N"])
    pests = float(obs.pest_risk)

    irrig = 0.0
    if m < 0.42:
        irrig = min(520.0, float(obs.water_available))
    elif m < 0.50:
        irrig = min(220.0, float(obs.water_available))

    fn = fp = fk = 0.0
    if n < 0.40 and day % 6 == 0:
        fn, fp, fk = 6.0, 4.5, 4.5
    elif n < 0.46 and day % 11 == 0:
        fn, fp, fk = 3.0, 2.5, 2.5

    pest = pests > 0.42 and day % 5 == 0

    return Action(
        irrigation=float(irrig),
        fertilizer_n=float(fn),
        fertilizer_p=float(fp),
        fertilizer_k=float(fk),
        pesticide=bool(pest),
        harvest=False,
    )


def main() -> None:
    env = FarmEnv()
    obs = env.reset(seed=42)
    print("=== reset ===")
    print(obs.model_dump())

    for i in range(10):
        act = _rule_based_policy(obs.day, obs)
        obs, rew, done, info = env.step(act)
        print(f"\n--- step {i + 1} | day_index={info['day']} | done={done} ---")
        print(f"reward={rew.value:.4f} | breakdown={ {k: round(v, 4) for k, v in rew.breakdown.items()} }")
        print(
            f"soil_moisture={obs.soil_moisture:.3f} | health={env.state()['crop_health']:.3f} "
            f"| stage={obs.crop_stage} | pests={obs.pest_risk:.3f}"
        )
        print(f"weather={info['weather']} | water_left={obs.water_available:.1f} | budget={obs.budget_remaining:.2f}")
        if done:
            print(f"terminated: {info.get('termination_reason')}")
            break


if __name__ == "__main__":
    main()

"""FastAPI server exposing FarmEnv for OpenEnv clients."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from openenv_farm.env.farm_env import FarmEnv
from openenv_farm.env.models import Action
from openenv_farm.graders.profit_grader import grade as grade_profit
from openenv_farm.graders.sustainability_grader import grade as grade_sustainability
from openenv_farm.graders.yield_grader import grade as grade_yield
from openenv_farm.tasks import easy, hard, medium

TASK_MODULES: dict[str, Any] = {
    "easy": easy,
    "medium": medium,
    "hard": hard,
}

env = FarmEnv()
current_task: str = "easy"

app = FastAPI(
    title="Smart Farm OpenEnv",
    version="1.0.0",
    root_path="/proxy/7860",
)


def _task_conds(name: str) -> dict[str, Any]:
    mod = TASK_MODULES.get(name)
    if mod is None:
        raise HTTPException(status_code=400, detail=f"Unknown task: {name}")
    return mod.get_initial_conditions()


def get_task_config(task: str) -> dict[str, Any]:
    mod = TASK_MODULES.get(task)
    if mod is None:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}")
    return mod.get_initial_conditions()


def handle_reset(task: str, seed: int | None) -> dict[str, Any]:
    global current_task
    current_task = task
    s = 42 if seed is None else int(seed)
    obs = env.reset(seed=s, task_config=get_task_config(task))
    return {"observation": obs.model_dump(), "task": current_task, "seed": s}


def handle_step(action: Action) -> dict[str, Any]:
    if env.done:
        raise HTTPException(status_code=400, detail="Episode finished; call /reset first")
    obs, rew, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": rew.model_dump(),
        "done": done,
        "info": info,
    }


def compute_grader() -> dict[str, Any]:
    hist = env.history
    conds = _task_conds(current_task)
    y = grade_yield(hist, task_name=current_task)
    p = grade_profit(hist, task_name=current_task)
    s = grade_sustainability(
        hist,
        task_name=current_task,
        max_water=float(conds["water_available"]),
        max_fertilizer=max(1.0, float(conds["budget"]) / 6.0),
    )
    return {
        "task": current_task,
        "yield_score": y,
        "profit_score": p,
        "sustainability_score": s,
    }


async def run_baseline_async() -> dict[str, Any]:
    from openenv_farm.inference import run_baseline

    try:
        return await asyncio.to_thread(run_baseline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def root() -> dict[str, Any]:
    return {"message": "Smart Farm OpenEnv API running"}


@app.get("/reset")
def reset_episode(
    task: str = Query("easy", description="easy | medium | hard"),
    seed: int | None = Query(None, description="RNG seed; default 42"),
) -> dict[str, Any]:
    return handle_reset(task, seed)


@app.get("/reset/")
def reset_episode_slash(
    task: str = Query("easy", description="easy | medium | hard"),
    seed: int | None = Query(None, description="RNG seed; default 42"),
) -> dict[str, Any]:
    return handle_reset(task, seed)


@app.post("/step")
def step_episode(action: Action) -> dict[str, Any]:
    return handle_step(action)


@app.post("/step/")
def step_episode_slash(action: Action) -> dict[str, Any]:
    return handle_step(action)


@app.get("/state")
def get_state() -> dict[str, Any]:
    return env.state()


@app.get("/state/")
def get_state_slash() -> dict[str, Any]:
    return env.state()


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": Action.model_json_schema(),
    }


@app.get("/tasks/")
def list_tasks_slash() -> dict[str, Any]:
    return list_tasks()


@app.get("/grader")
def run_grader() -> dict[str, Any]:
    return compute_grader()


@app.get("/grader/")
def run_grader_slash() -> dict[str, Any]:
    return compute_grader()


@app.get("/baseline")
async def baseline() -> dict[str, Any]:
    """Runs baseline inference off the event loop (may take several minutes)."""
    return await run_baseline_async()


@app.get("/baseline/")
async def baseline_slash() -> dict[str, Any]:
    return await run_baseline_async()

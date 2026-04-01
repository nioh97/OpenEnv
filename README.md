
---
title: Smart Farm OpenEnv
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Smart Farming OpenEnv Environment

## Overview

Real-world reinforcement-learning environment for **seasonal field operations**: irrigation, NPK fertilization, pest control, and harvest timing under **water, budget, and weather** constraints. Suited for training or evaluating decision-making agents (RL, LLM policies, or heuristics).

## Observation space

Each step exposes a typed **Observation** (Pydantic), including:

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Episode day index |
| `soil_moisture` | float | 0–1 |
| `soil_nutrients` | dict | `N`, `P`, `K` each in 0–1 |
| `crop_stage` | str | `sowing`, `vegetative`, `flowering`, `mature` |
| `pest_risk` | float | 0–1 |
| `water_available` | float | Liters remaining |
| `budget_remaining` | float | Currency for inputs |
| `weather_forecast` | list | Next 3 days: `temperature`, `rainfall` per day |

Full schema: `GET /tasks` → `action_schema` / OpenAPI, or `Observation` in `openenv_farm/env/models.py`.

## Action space

Each step accepts an **Action** (JSON / Pydantic):

| Field | Type | Description |
|-------|------|-------------|
| `irrigation` | float | Liters (≥ 0) |
| `fertilizer_n` | float | kg N (≥ 0) |
| `fertilizer_p` | float | kg P (≥ 0) |
| `fertilizer_k` | float | kg K (≥ 0) |
| `pesticide` | bool | Apply pesticide |
| `harvest` | bool | End episode with harvest (must be the only operation that day) |

## Tasks (easy → hard)

| Task | Intent |
|------|--------|
| **easy** | Higher starting moisture, nutrients, water, budget; lower initial pest pressure. |
| **medium** | Moderate resources and pests. |
| **hard** | Tighter water/budget and lower starting soil quality; higher pest pressure. |

Episode graders (deterministic, in `[0, 1]`): **yield**, **profit**, **sustainability** — see `openenv_farm/graders/`.

## Reward

Dense shaped reward per step (health, growth, costs, stress, pests, harvest, etc.) with a **breakdown** dict for debugging. Implemented in `openenv_farm/env/reward.py` (environment core).

## API (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/reset?task=easy&seed=42` | Reset env with task + optional seed |
| POST | `/step` | Body: Action JSON → observation, reward, done, info |
| GET | `/state` | Full internal state dict |
| GET | `/tasks` | Task list + `Action` JSON schema |
| GET | `/grader` | Yield / profit / sustainability scores for current episode history |
| GET | `/baseline` | Runs full baseline (async thread); may take minutes |

Interactive API docs (Swagger UI): **`/docs`** — locally `http://127.0.0.1:7860/docs`; on Hugging Face Spaces use your Space URL with the proxy prefix, e.g. `https://<your-space>.hf.space/proxy/7860/docs`.

### Example `curl` requests

**Local** (after `uvicorn` on port 7860):

```bash
curl -s http://127.0.0.1:7860/
curl -s "http://127.0.0.1:7860/reset?task=easy"
curl -s http://127.0.0.1:7860/state
curl -s http://127.0.0.1:7860/tasks
curl -s http://127.0.0.1:7860/grader
curl -s -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"irrigation":0,"fertilizer_n":0,"fertilizer_p":0,"fertilizer_k":0,"pesticide":false,"harvest":false}'
```

**Hugging Face Spaces** (replace `BASE` with your Space API base, including `/proxy/7860`):

```bash
BASE="https://<your-space>.hf.space/proxy/7860"
curl -s "$BASE/"
curl -s "$BASE/reset?task=easy"
curl -s "$BASE/state"
```

## Local setup

```bash
cd /path/to/OpenEnv   # repository root (parent of openenv_farm)
pip install -r openenv_farm/requirements.txt
uvicorn openenv_farm.api.server:app --host 127.0.0.1 --port 7860
```

Smoke test: open **`http://127.0.0.1:7860/docs`** (FastAPI Swagger) or run `curl -s http://127.0.0.1:7860/` and `curl -s "http://127.0.0.1:7860/reset?task=easy"`.

## Baseline inference (`inference.py`)

Run from **repository root** (required for imports):

```bash
python inference.py
```

Environment variables:

| Variable | Role |
|----------|------|
| `OPENAI_API_KEY` | Primary API key for OpenAI-compatible chat API |
| `HF_TOKEN` | Used if `OPENAI_API_KEY` is unset (e.g. Hugging Face Inference) |
| `API_BASE_URL` | Optional custom base URL |
| `MODEL_NAME` | Model id (default `gpt-4o-mini`) |

With **no** key set, the baseline uses the **heuristic** policy only (still produces scores). LLM path uses **temperature 0** for reproducibility; provider-side nondeterminism may still vary slightly.

Output JSON includes `easy`, `medium`, `hard` mean grader scores plus a `meta` block (steps, `openai` vs `heuristic`). Target runtime: under ~20 minutes on 2 vCPU / 8 GB when using the LLM.

## Docker

```bash
docker build -t smart-farm-openenv .
docker run -p 7860:7860 smart-farm-openenv
```

API on port **7860**. Image includes `inference.py` at `/app/inference.py`; run `docker run ... python inference.py` for baseline inside the container.

## OpenEnv metadata

See `openenv_farm/openenv.yaml`. Validate with:

```bash
pip install openenv-core
openenv validate
```

(Adjust if your `openenv` CLI expects a different layout.)

## Deployment

Docker-based; compatible with **Hugging Face Spaces** (Docker SDK, port 7860). Tag the Space with **openenv** per platform instructions.

## Goal

Train and evaluate agents for **agricultural decision-making** under uncertainty and resource limits.

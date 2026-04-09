from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rl.gym_env import SmartFactoryEnv
from .tasks import TASK_LIST, get_task, serialize_tasks

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except Exception:
    gr = None
    GRADIO_AVAILABLE = False


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Any = 0


app = FastAPI(title="Smart Factory OpenEnv API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[SmartFactoryEnv] = None
_current_task_id = TASK_LIST[0]["id"]
_current_observation: Optional[np.ndarray] = None


def _observation_to_json(observation: Any) -> Any:
    if isinstance(observation, np.ndarray):
        return observation.tolist()
    if isinstance(observation, (np.floating, np.integer)):
        return observation.item()
    return observation


def _ensure_env(task_id: Optional[str] = None) -> SmartFactoryEnv:
    global _env, _current_task_id

    requested_task = task_id or _current_task_id
    task = get_task(requested_task)
    _current_task_id = task["id"]
    if _env is None or _env.task_id != _current_task_id:
        _env = SmartFactoryEnv(task_id=_current_task_id, render_mode=None)
    return _env


def _current_state() -> Dict[str, Any]:
    env = _ensure_env()
    return env.sim.get_state()


def _encode_action(action: Any) -> int:
    state = _current_state()
    machines = state.get("machines", [])
    queue = state.get("job_queue", [])

    if isinstance(action, bool):
        return int(action)
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, dict):
        if isinstance(action.get("action"), (int, np.integer)):
            return int(action["action"])

        machine = action.get("machine")
        task = action.get("task")
        if isinstance(machine, int) and isinstance(task, int):
            if machine < 0 or task < 0:
                return 0
            return 19 + (task * 8) + machine

        action_type = str(action.get("type", "do_nothing"))
        if action_type == "do_nothing":
            return 0
        if action_type == "maintenance":
            machine_id = action.get("machine_id")
            for idx, machine_state in enumerate(machines[:8]):
                if machine_state.get("id") == machine_id:
                    return 1 + idx
            return 0
        if action_type == "delay":
            job_id = action.get("job_id")
            for idx, job_state in enumerate(queue[:10]):
                if job_state.get("id") == job_id:
                    return 9 + idx
            return 0
        if action_type == "assign":
            job_id = action.get("job_id")
            machine_id = action.get("machine_id")
            job_idx = next((idx for idx, job_state in enumerate(queue[:10]) if job_state.get("id") == job_id), None)
            machine_idx = next(
                (idx for idx, machine_state in enumerate(machines[:8]) if machine_state.get("id") == machine_id),
                None,
            )
            if job_idx is None or machine_idx is None:
                return 0
            return 19 + (job_idx * 8) + machine_idx

    return 0


@app.post("/reset")
def reset(request: Optional[ResetRequest] = Body(default=None)) -> Dict[str, Any]:
    global _current_observation

    try:
        env = _ensure_env(request.task_id if request else None)
        observation, _ = env.reset()
        _current_observation = observation
        return {"observation": _observation_to_json(observation)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    global _current_observation

    try:
        env = _ensure_env()
        if _current_observation is None:
            _current_observation, _ = env.reset()

        action_value = _encode_action(request.action)
        observation, reward, terminated, truncated, info = env.step(action_value)
        _current_observation = observation
        return {
            "observation": _observation_to_json(observation),
            "reward": float(max(0.01, min(0.99, float(reward)))),
            "done": bool(terminated or truncated),
            "info": info if isinstance(info, dict) else {},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
def state() -> Dict[str, Any]:
    return {"observation": _observation_to_json(_current_observation), "state": _current_state()}


@app.get("/tasks")
def tasks() -> list[Dict[str, Any]]:
    return serialize_tasks()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "task_id": _current_task_id}


if GRADIO_AVAILABLE:
    with gr.Blocks(title="Smart Factory Demo") as demo:
        gr.Markdown("# AI-Powered Smart Factory Optimization")
        gr.Markdown("Use POST /reset and POST /step for OpenEnv validation. This mounted page is optional.")
        gr.JSON(value={"available_tasks": [task["id"] for task in TASK_LIST]})

    app = gr.mount_gradio_app(app, demo, path="/demo")

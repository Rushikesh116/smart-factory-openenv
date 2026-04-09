import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks import TASK_LIST, get_task, grade_task, serialize_tasks
from rl.gym_env import SmartFactoryEnv as GymSmartFactoryEnv

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MaskablePPO = None
    MASKABLE_PPO_AVAILABLE = False

from stable_baselines3 import PPO


app = FastAPI(title="Smart Factory RL API")
model = None
env: Optional[GymSmartFactoryEnv] = None
current_observation: Optional[np.ndarray] = None
current_task_id = TASK_LIST[0]["id"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ActionRequest(BaseModel):
    action: Dict[str, Any]


class AutoRunRequest(BaseModel):
    max_steps: int = 50


def load_model() -> None:
    global model
    model_path = "models/latest.zip"
    if not os.path.exists(model_path):
        model = None
        return

    if MASKABLE_PPO_AVAILABLE:
        try:
            model = MaskablePPO.load(model_path)
            return
        except Exception:
            pass

    try:
        model = PPO.load(model_path)
    except Exception:
        model = None


def _ensure_env(task_id: Optional[str] = None) -> GymSmartFactoryEnv:
    global env, current_task_id

    requested_task = task_id or current_task_id
    task = get_task(requested_task)
    current_task_id = task["id"]
    if env is None or env.task_id != current_task_id:
        env = GymSmartFactoryEnv(task_id=current_task_id)
    return env


def _current_state() -> Dict[str, Any]:
    active_env = _ensure_env()
    return active_env.sim.get_state()


def _encode_action(action: Dict[str, Any], state: Dict[str, Any]) -> int:
    if isinstance(action.get("action"), int):
        return int(action["action"])

    machine = action.get("machine")
    task = action.get("task")
    if isinstance(machine, int) and isinstance(task, int):
        if machine < 0 or task < 0:
            return 0
        return 19 + (task * 8) + machine

    action_type = str(action.get("type", "do_nothing"))
    machines = state.get("machines", [])
    queue = state.get("job_queue", [])

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


def _step_with_action(action_value: int) -> Dict[str, Any]:
    global current_observation

    active_env = _ensure_env()
    if current_observation is None:
        current_observation, _ = active_env.reset()

    obs, reward, terminated, truncated, info = active_env.step(action_value)
    current_observation = obs
    return {
        "action": int(action_value),
        "observation": obs.tolist() if hasattr(obs, "tolist") else obs,
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "info": info,
    }


@app.on_event("startup")
async def startup_event() -> None:
    load_model()
    _ensure_env(current_task_id)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <h1>Smart Factory RL System</h1>
    <p>Status: <b>Running</b></p>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/docs">Swagger UI</a></li>
        <li>POST /reset</li>
        <li>POST /step</li>
        <li>POST /predict</li>
        <li>GET /tasks</li>
        <li>GET /grader</li>
    </ul>
    """


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "task_id": current_task_id, "model_loaded": model is not None}



@app.get("/reset")
def reset_get() -> Dict[str, Any]:
    return {"status": "ok", "message": "Use POST /reset to initialize environment"}

@app.get("/step")
def step_get() -> Dict[str, Any]:
    return {"status": "ok", "message": "Use POST /step with action payload"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global current_observation

    active_env = _ensure_env(request.task_id if request else None)
    try:
        obs, info = active_env.reset()
        current_observation = obs
        return {
            "observation": obs.tolist() if hasattr(obs, "tolist") else obs,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {exc}") from exc


@app.get("/state")
def state() -> Dict[str, Any]:
    try:
        return {"task_id": current_task_id, "state": _current_state()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read state: {exc}") from exc


@app.get("/tasks")
def get_tasks() -> list[Dict[str, Any]]:
    return serialize_tasks()


@app.get("/grader")
def get_grader() -> Dict[str, Any]:
    try:
        state_data = _current_state()
        raw_score = float(grade_task(current_task_id, state_data))

        # STRICT VALIDATOR SAFE CLAMP
        if raw_score <= 0.0:
            score = 0.01
        elif raw_score >= 1.0:
            score = 0.99
        else:
            score = raw_score

        return {
            "task_id": current_task_id,
            "score": score
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to grade environment: {exc}"
        ) from exc


@app.post("/step")
def step(request: ActionRequest) -> Dict[str, Any]:
    try:
        state_data = _current_state()
        action_value = _encode_action(request.action, state_data)
        result = _step_with_action(action_value)
        result["state"] = _current_state()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to execute step: {exc}") from exc


@app.post("/predict")
def predict() -> Dict[str, Any]:
    global current_observation

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    active_env = _ensure_env()
    if current_observation is None:
        current_observation, _ = active_env.reset()

    try:
        action, _ = model.predict(current_observation, deterministic=True)
        action_value = int(action[0]) if isinstance(action, np.ndarray) else int(action)
        result = _step_with_action(action_value)
        result["state"] = _current_state()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to predict and step: {exc}") from exc


@app.post("/auto-run")
def auto_run(request: AutoRunRequest) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = []
    steps_taken = 0
    done = False

    try:
        _ensure_env()
        while not done and steps_taken < request.max_steps:
            result = predict()
            results.append(result)
            done = bool(result["done"])
            steps_taken += 1
        return {"total_steps": steps_taken, "results": results, "final_done": done}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run auto steps: {exc}") from exc


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from app.tasks import TASKS
from rl.gym_env import SmartFactoryEnv as GymSmartFactoryEnv
import numpy as np
from typing import Dict, Any, Optional

# Try to import both PPO types
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False

from stable_baselines3 import PPO

# Global variables
app = FastAPI(title="Smart Factory RL API")
model = None
env: Optional[GymSmartFactoryEnv] = None
current_observation: Optional[np.ndarray] = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """Load the trained model (try MaskablePPO first, then PPO)"""
    global model
    model_path = "models/latest.zip"

    try:
        if MASKABLE_PPO_AVAILABLE:
            try:
                model = MaskablePPO.load(model_path)
                print(f"✅ MaskablePPO model loaded from {model_path}")
                return
            except Exception as e:
                print(f"MaskablePPO failed: {e}")

        # Fallback to regular PPO
        model = PPO.load(model_path)
        print(f"✅ PPO model loaded from {model_path}")

    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        print("💡 Make sure you have trained a model first with: python -m rl.train")
        print("💡 If using MaskablePPO, ensure sb3-contrib version compatibility")
        model = None

def initialize_environment():
    """Initialize the Smart Factory gym environment"""
    global env
    try:
        # Use gym environment for proper observation formatting
        env = GymSmartFactoryEnv(task_id="easy")
        print("✅ Gym environment initialized")
    except Exception as e:
        print(f"❌ Failed to initialize gym environment: {e}")
        env = None

@app.on_event("startup")
async def startup_event():
    """Load model and initialize environment on startup"""
    load_model()
    initialize_environment()

@app.get("/", response_class=HTMLResponse)
def root():
    """Simple HTML status page"""
    return """
    <h1>🏭 Smart Factory RL System</h1>
    <p>Status: <b>Running ✅</b></p>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/docs">Swagger UI</a></li>
        <li>POST /reset</li>
        <li>POST /step</li>
        <li>POST /predict</li>
    </ul>
    """

@app.post("/reset")
def reset():
    """Reset the environment and return initial observation"""
    global current_observation

    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        obs, info = env.reset()
        current_observation = obs
        print(f"DEBUG FASTAPI RESET: obs type={type(obs)}, shape={obs.shape if hasattr(obs, 'shape') else 'no shape'}")
        return {"observation": obs.tolist() if hasattr(obs, 'tolist') else obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")

class ActionRequest(BaseModel):
    action: Dict[str, Any]

@app.post("/step")
def step(request: ActionRequest):
    """Take a step in the environment"""
    global current_observation

    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        # Convert action dict to the format expected by gym env
        # The gym env expects discrete actions, so we need to encode the machine/task back to discrete
        machine_idx = request.action.get("machine", 0)
        task_idx = request.action.get("task", 0)

        # For now, assume the action dict contains the discrete action directly
        # This is a simplification - in practice you'd need proper encoding
        if isinstance(request.action.get("action"), int):
            action = request.action["action"]
        else:
            # Default to action 0 (do nothing)
            action = 0

        obs, reward, done, truncated, info = env.step(action)
        current_observation = obs
        return {
            "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
            "reward": float(reward),
            "done": bool(done or truncated),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute step: {str(e)}")

@app.post("/predict")
def predict():
    """Use trained model to predict and execute next action"""
    global current_observation

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    if current_observation is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first")

    try:
        print(f"DEBUG FASTAPI PREDICT: current_observation type={type(current_observation)}, shape={current_observation.shape if hasattr(current_observation, 'shape') else 'no shape'}")
        # Get action from model
        action, _ = model.predict(current_observation, deterministic=True)

        # Action is already a discrete value from the model's action space
        action_value = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        # Execute step with discrete action
        obs, reward, done, truncated, info = env.step(action_value)
        current_observation = obs

        return {
            "action": action_value,
            "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
            "reward": float(reward),
            "done": bool(done or truncated),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict and step: {str(e)}")

# Step endpoint for manual action
class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.post("/step")
def step_manual(request: StepRequest):
    """Execute a manual step with provided action"""
    global current_observation

    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    if current_observation is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first")

    try:
        action_dict = request.action
        machine = action_dict.get("machine", 0)
        task = action_dict.get("task", 0)

        # Convert to discrete action (same logic as gym_env.py)
        # Action encoding: 0 = do_nothing, 1-9 = assign task 0-8 to machine 0, etc.
        if machine == -1 or task == -1:  # do_nothing
            action_value = 0
        else:
            action_value = 1 + machine * 9 + task  # 1-81 for assignments

        # Execute step
        obs, reward, done, truncated, info = env.step(action_value)
        current_observation = obs

        return {
            "action": action_value,
            "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
            "reward": float(reward),
            "done": bool(done or truncated),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to step: {str(e)}")

# Optional auto-run endpoint
class AutoRunRequest(BaseModel):
    max_steps: int = 50

@app.post("/auto-run")
def auto_run(request: AutoRunRequest):
    """Run multiple steps using model until done or max_steps reached"""
    global current_observation

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    if current_observation is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first")

    results = []
    steps_taken = 0

    try:
        while not env.done and steps_taken < request.max_steps:
            # Get action from model
            action, _ = model.predict(current_observation, deterministic=True)

            # Action is already discrete
            action_value = int(action[0]) if isinstance(action, np.ndarray) else int(action)

            # Execute step
            obs, reward, done, truncated, info = env.step(action_value)
            current_observation = obs

            results.append({
                "step": steps_taken + 1,
                "action": action_value,
                "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
                "reward": float(reward),
                "done": bool(done or truncated),
                "info": info
            })

            steps_taken += 1

            if done or truncated:
                break

        return {
            "total_steps": steps_taken,
            "results": results,
            "final_done": bool(env.done)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run auto steps: {str(e)}")

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
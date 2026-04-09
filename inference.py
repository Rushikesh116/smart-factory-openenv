from __future__ import annotations

import os
from typing import Any

import numpy as np

from rl.baselines import HeuristicBaselinePolicy
from rl.gym_env import SmartFactoryEnv

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MaskablePPO = None
    MASKABLE_PPO_AVAILABLE = False

from stable_baselines3 import PPO


MODEL_NAME = os.getenv("MODEL_NAME", "smart-factory-rl")
TASK_ID = os.getenv("TASK_ID", "easy")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "latest.zip"))


def load_policy(model_path: str) -> tuple[Any | None, str]:
    if not os.path.exists(model_path):
        return None, "heuristic"

    if MASKABLE_PPO_AVAILABLE:
        try:
            return MaskablePPO.load(model_path), "maskable_ppo"
        except Exception:
            pass

    try:
        return PPO.load(model_path), "ppo"
    except Exception:
        return None, "heuristic"


def choose_action(env: SmartFactoryEnv, model: Any | None, policy_kind: str, fallback: HeuristicBaselinePolicy, obs: Any) -> tuple[int, str]:
    if model is None:
        action = int(fallback.act(env, obs))
        return action, "heuristic"

    if policy_kind == "maskable_ppo":
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    else:
        action, _ = model.predict(obs, deterministic=True)

    if isinstance(action, np.ndarray):
        action_value = int(action.item()) if action.ndim == 0 else int(action[0])
    else:
        action_value = int(action)
    return action_value, policy_kind


def main() -> None:
    task_name = f"smart-factory-{TASK_ID}"
    env_name = "smart-factory-openenv"
    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

    env = SmartFactoryEnv(task_id=TASK_ID, render_mode=None)
    model, policy_kind = load_policy(MODEL_PATH)
    fallback_policy = HeuristicBaselinePolicy()

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    reward_trace: list[str] = []
    max_steps = max(10, int(env.config.get("max_steps", 50)))

    while not done and steps < max_steps:
        action, action_label = choose_action(env, model, policy_kind, fallback_policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        total_reward += float(reward)
        steps += 1
        reward_trace.append(f"{float(reward):.2f}")
        print(
            f"[STEP] step={steps} action={action_label}:{action} "
            f"reward={float(reward):.2f} done={str(done).lower()} error=null"
        )

    success = "true" if steps > 0 else "false"
    rewards = ",".join(reward_trace) if reward_trace else f"{total_reward:.2f}"
    print(f"[END] success={success} steps={steps} rewards={rewards}")


if __name__ == "__main__":
    main()

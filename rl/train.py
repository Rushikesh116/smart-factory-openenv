from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime
from typing import Callable

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from rl.config import PPOConfig, TrainConfig
from rl.gym_env import SmartFactoryEnv
import rl.gym_env as gym_env_module
from rl.utils import copy_if_exists, ensure_dir

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_RL_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
RUNS_DIR = os.path.join(_PROJECT_ROOT, "runs")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return _schedule


def constant_schedule(value: float) -> Callable[[float], float]:
    def _schedule(_: float) -> float:
        return value

    return _schedule


def _tensorboard_log_dir(path: str) -> str:
    if importlib.util.find_spec("tensorboard") is None:
        raise ImportError("TensorBoard is required. Install it with 'pip install tensorboard'.")
    return ensure_dir(path)


def _create_run_dir(run_id: str | None = None) -> tuple[str, str]:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(RUNS_DIR, f"run_{run_id}"))
    return run_id, run_dir


def _make_vec_env(task_id: str, render_mode=None):
    def _factory():
        return SmartFactoryEnv(task_id=task_id, render_mode=render_mode)

    env = DummyVecEnv([_factory])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def _load_eval_npz_summary(eval_log_dir: str) -> dict:
    eval_npz_path = os.path.join(eval_log_dir, "evaluations.npz")
    if not os.path.exists(eval_npz_path):
        return {
            "eval_npz_path": eval_npz_path,
            "eval_timesteps": [],
            "eval_reward_curve": [],
            "best_eval_reward": None,
            "final_eval_reward": None,
        }

    data = np.load(eval_npz_path)
    timesteps = data["timesteps"].tolist() if "timesteps" in data else []
    results = np.asarray(data["results"], dtype=np.float32) if "results" in data else np.asarray([], dtype=np.float32)
    reward_curve = results.mean(axis=1).astype(float).tolist() if results.size else []
    return {
        "eval_npz_path": eval_npz_path,
        "eval_timesteps": timesteps,
        "eval_reward_curve": reward_curve,
        "best_eval_reward": float(max(reward_curve)) if reward_curve else None,
        "final_eval_reward": float(reward_curve[-1]) if reward_curve else None,
    }


def _write_training_summary(
    *,
    stage_log_dir: str,
    task_id: str,
    requested_timesteps: int,
    elapsed_seconds: float,
    stage_model_zip: str,
    stage_stats_path: str,
    best_model_path: str,
    eval_log_dir: str,
    post_eval_summary: dict | None = None,
) -> str:
    eval_summary = _load_eval_npz_summary(eval_log_dir)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "task_id": task_id,
        "requested_timesteps": int(requested_timesteps),
        "wall_clock_seconds": float(elapsed_seconds),
        "steps_per_second": float(requested_timesteps / max(elapsed_seconds, 1e-6)),
        "artifacts": {
            "stage_model": stage_model_zip,
            "stage_stats": stage_stats_path,
            "best_model": best_model_path,
            "eval_log_dir": eval_log_dir,
        },
        "evaluation": eval_summary,
    }
    if post_eval_summary is not None:
        summary["post_training_policy_comparison"] = post_eval_summary

    summary_path = os.path.join(stage_log_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary_path


class RewardDebugCallback(BaseCallback):
    def __init__(self, log_every: int = 256, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = max(1, int(log_every))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.logger.record("custom/step_reward", float(np.mean(np.asarray(rewards, dtype=np.float32))))

        if self.n_calls % self.log_every == 0:
            print("DebugCallback rewards:", rewards)

        for info in infos:
            if not isinstance(info, dict):
                continue
            rb = info.get("reward_breakdown")
            if isinstance(rb, dict):
                for key, value in rb.items():
                    self.logger.record(f"reward/{key}", float(value))
            if "completion_rate" in info:
                self.logger.record("custom/completion_rate", float(info["completion_rate"]))
            if "action_validity_rate" in info:
                self.logger.record("custom/action_validity_rate", float(info["action_validity_rate"]))
            episode = info.get("episode")
            if episode:
                self.logger.record("custom/episode_reward", float(episode["r"]))
                self.logger.record("custom/episode_length", float(episode["l"]))
                break
        return True


def train_ppo(
    timesteps: int = 20000,
    task_id: str = "medium",
    eval_freq: int = 5000,
    checkpoint_freq: int = 5000,
    load_model_path: str | None = None,
    load_stats_path: str | None = None,
    reset_num_timesteps: bool = True,
    run_dir: str | None = None,
    debug_env: bool = False,
    post_eval_episodes: int = 0,
) -> dict:
    print(f"Starting PPO training | task={task_id} | timesteps={timesteps}")
    if run_dir is None:
        _, run_dir = _create_run_dir()
    stage_log_dir = ensure_dir(os.path.join(run_dir, task_id))
    tensorboard_dir = _tensorboard_log_dir(stage_log_dir)
    best_model_dir = ensure_dir(os.path.join(stage_log_dir, "best_model"))
    eval_log_dir = ensure_dir(os.path.join(stage_log_dir, "eval"))
    checkpoint_dir = ensure_dir(os.path.join(stage_log_dir, "checkpoints"))
    gym_env_module.DEBUG = bool(debug_env)
    ensure_dir(MODELS_DIR)

    env = _make_vec_env(task_id=task_id, render_mode=None)
    if load_stats_path and os.path.exists(load_stats_path):
        loaded_norm = VecNormalize.load(load_stats_path, env.venv)
        loaded_norm.training = True
        loaded_norm.norm_reward = True
        env = loaded_norm

    eval_env = _make_vec_env(task_id=task_id, render_mode=None)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    ppo_cfg = PPOConfig()
    learning_rate = (
        linear_schedule(ppo_cfg.learning_rate)
        if ppo_cfg.use_linear_lr_schedule
        else ppo_cfg.learning_rate
    )
    clip_range_schedule = constant_schedule(ppo_cfg.clip_range)
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=list(ppo_cfg.pi_layers), vf=list(ppo_cfg.vf_layers)),
    )
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading model weights from: {load_model_path}")
        model = MaskablePPO.load(load_model_path, env=env, tensorboard_log=tensorboard_dir, device="auto")
        model.learning_rate = learning_rate
        model.n_steps = ppo_cfg.n_steps
        model.batch_size = ppo_cfg.batch_size
        model.n_epochs = ppo_cfg.n_epochs
        model.gamma = ppo_cfg.gamma
        model.gae_lambda = ppo_cfg.gae_lambda
        model.clip_range = clip_range_schedule
        model.ent_coef = ppo_cfg.ent_coef
        model.vf_coef = ppo_cfg.vf_coef
        model.max_grad_norm = ppo_cfg.max_grad_norm
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            clip_range=clip_range_schedule,
            ent_coef=ppo_cfg.ent_coef,
            vf_coef=ppo_cfg.vf_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_dir,
        )

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, min_evals=5, verbose=1)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, checkpoint_freq),
        save_path=checkpoint_dir,
        name_prefix=f"ppo_{task_id}",
        save_vecnormalize=True,
    )

    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([RewardDebugCallback(log_every=256), checkpoint_callback, eval_callback]),
        tb_log_name=f"PPO_smart_factory_{task_id}",
        log_interval=1,
        reset_num_timesteps=reset_num_timesteps,
    )
    elapsed = time.time() - start_time

    stage_model_path = os.path.join(stage_log_dir, f"{task_id}_model")
    stage_model_zip = f"{stage_model_path}.zip"
    stage_stats_path = os.path.join(stage_log_dir, "vecnormalize.pkl")
    model.save(stage_model_path)
    env.save(stage_stats_path)

    latest_model = os.path.join(MODELS_DIR, "latest.zip")
    latest_stats = os.path.join(MODELS_DIR, "latest_vecnormalize.pkl")
    copy_if_exists(stage_model_zip, latest_model)
    copy_if_exists(stage_stats_path, latest_stats)

    post_eval_summary = None
    if post_eval_episodes > 0:
        try:
            from rl.evaluate import evaluate_policies

            evaluation_dir = ensure_dir(os.path.join(stage_log_dir, "evaluation"))
            post_eval_summary = evaluate_policies(
                model_path=stage_model_zip,
                task_id=task_id,
                num_episodes=post_eval_episodes,
                stats_path=stage_stats_path,
                label=f"{task_id}_post_train",
                include_timestamp=False,
                output_dir=evaluation_dir,
            )
        except Exception as exc:
            post_eval_summary = {"error": str(exc)}

    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    training_summary_path = _write_training_summary(
        stage_log_dir=stage_log_dir,
        task_id=task_id,
        requested_timesteps=timesteps,
        elapsed_seconds=elapsed,
        stage_model_zip=stage_model_zip,
        stage_stats_path=stage_stats_path,
        best_model_path=best_model_path,
        eval_log_dir=eval_log_dir,
        post_eval_summary=post_eval_summary,
    )

    print(f"Training completed in {elapsed:.2f}s")
    print(f"Saved stage model: {stage_model_zip}")
    print(f"Saved stage vecnorm: {stage_stats_path}")
    print(f"Updated latest model: {latest_model}")
    print(f"Best model (eval callback): {best_model_path}")
    print(f"Training summary: {training_summary_path}")
    paths = {
        "task_id": task_id,
        "run_dir": run_dir,
        "stage_dir": stage_log_dir,
        "stage_model": stage_model_zip,
        "stage_stats": stage_stats_path,
        "latest_model": latest_model,
        "latest_stats": latest_stats,
        "best_model": best_model_path,
        "training_summary": training_summary_path,
    }
    if post_eval_summary is not None:
        paths["post_eval_summary"] = post_eval_summary
    return paths


def train_curriculum(run_dir: str | None = None, debug_env: bool = False, post_eval_episodes: int = 0) -> dict:
    defaults = TrainConfig()
    stages = [
        ("easy", defaults.curriculum_easy_timesteps),
        ("medium", defaults.curriculum_medium_timesteps),
        ("hard", defaults.curriculum_hard_timesteps),
    ]

    if run_dir is None:
        _, run_dir = _create_run_dir()
    latest_model_path = None
    latest_stats_path = None
    final_paths = {}
    for i, (task_id, timesteps) in enumerate(stages):
        print(f"\n=== Curriculum stage {i + 1}/{len(stages)}: {task_id} ({timesteps} timesteps) ===")
        final_paths = train_ppo(
            timesteps=timesteps,
            task_id=task_id,
            eval_freq=max(1000, min(5000, timesteps // 4)),
            checkpoint_freq=max(1000, min(5000, timesteps // 4)),
            load_model_path=latest_model_path,
            load_stats_path=latest_stats_path,
            reset_num_timesteps=(i == 0),
            run_dir=run_dir,
            debug_env=debug_env,
            post_eval_episodes=(post_eval_episodes if i == len(stages) - 1 else 0),
        )
        latest_model_path = final_paths["stage_model"]
        latest_stats_path = final_paths["stage_stats"]
    return final_paths


def _parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train PPO on SmartFactoryEnv")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "curriculum"])
    parser.add_argument("--task-id", type=str, default=defaults.task_id, choices=["easy", "medium", "hard"])
    parser.add_argument("--timesteps", type=int, default=defaults.timesteps)
    parser.add_argument("--eval-freq", type=int, default=defaults.eval_freq)
    parser.add_argument("--checkpoint-freq", type=int, default=defaults.checkpoint_freq)
    parser.add_argument("--load-model-path", type=str, default=None)
    parser.add_argument("--load-stats-path", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id used under runs/run_<id>")
    parser.add_argument("--debug-env", action="store_true", help="Enable verbose env debug prints during training.")
    parser.add_argument(
        "--post-eval-episodes",
        type=int,
        default=0,
        help="Optional number of post-training episodes for PPO/random/heuristic comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _, run_dir = _create_run_dir(args.run_id)
    if args.mode == "curriculum":
        train_curriculum(
            run_dir=run_dir,
            debug_env=args.debug_env,
            post_eval_episodes=args.post_eval_episodes,
        )
    else:
        train_ppo(
            timesteps=args.timesteps,
            task_id=args.task_id,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            load_model_path=args.load_model_path,
            load_stats_path=args.load_stats_path,
            run_dir=run_dir,
            debug_env=args.debug_env,
            post_eval_episodes=args.post_eval_episodes,
        )

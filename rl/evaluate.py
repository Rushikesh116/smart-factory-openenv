import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.gym_env import SmartFactoryEnv
import rl.gym_env as gym_env_module

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_RL_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")


def _make_eval_env(task_id: str):
    env = DummyVecEnv([lambda: SmartFactoryEnv(task_id=task_id, render_mode=None)])
    env = VecMonitor(env)
    return env


def _ensure_plots_dir() -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return PLOTS_DIR


def _save_metric_plot(values: list[float], title: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(values, marker="o", linewidth=1.6, markersize=3)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close()


def _plot_eval_curves(per_episode: dict[str, list[float]], label: str, include_timestamp: bool) -> dict[str, str]:
    plots_dir = _ensure_plots_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else None
    suffix = f"_{label}" if label else ""
    if ts:
        suffix = f"{suffix}_{ts}"

    paths = {
        "reward_curve": os.path.join(plots_dir, f"reward_curve{suffix}.png"),
        "completion_curve": os.path.join(plots_dir, f"completion_curve{suffix}.png"),
        "throughput_curve": os.path.join(plots_dir, f"throughput_curve{suffix}.png"),
        "idle_machines_curve": os.path.join(plots_dir, f"idle_machines_curve{suffix}.png"),
        "results_json": os.path.join(plots_dir, f"eval_results{suffix}.json"),
    }

    _save_metric_plot(per_episode["rewards"], "Reward per Episode", "Reward", paths["reward_curve"])
    _save_metric_plot(
        per_episode["completion_rates"],
        "Completion Rate per Episode",
        "Completion Rate",
        paths["completion_curve"],
    )
    _save_metric_plot(
        per_episode["throughputs"],
        "Throughput (Jobs per Step)",
        "Throughput",
        paths["throughput_curve"],
    )
    _save_metric_plot(
        per_episode["idle_machines_list"],
        "Average Idle Machines",
        "Idle Machines",
        paths["idle_machines_curve"],
    )
    with open(paths["results_json"], "w", encoding="utf-8") as f:
        json.dump(per_episode, f, indent=2)
    return paths


def _plot_overlay_from_json(current: dict[str, list[float]], baseline_json: str, label: str, include_timestamp: bool) -> str | None:
    if not baseline_json:
        return None
    if not os.path.exists(baseline_json):
        print(f"Overlay skipped; baseline json not found: {baseline_json}")
        return None

    with open(baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    plots_dir = _ensure_plots_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else None
    suffix = f"_{label}" if label else ""
    if ts:
        suffix = f"{suffix}_{ts}"
    path = os.path.join(plots_dir, f"reward_overlay{suffix}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(baseline.get("rewards", []), label="baseline", linewidth=1.6)
    plt.plot(current.get("rewards", []), label=label or "current", linewidth=1.6)
    plt.title("Reward Comparison (A/B)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def evaluate_agent(
    model_path: str,
    task_id: str = "easy",
    num_episodes: int = 10,
    stats_path: str | None = None,
    label: str = "experiment",
    include_timestamp: bool = False,
    overlay_json: str | None = None,
):
    gym_env_module.DEBUG = False
    print(f"\nEvaluating PPO agent | task={task_id} | episodes={num_episodes}")
    env = _make_eval_env(task_id)

    if stats_path and os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats: {stats_path}")
    else:
        print("No VecNormalize stats loaded; evaluating on raw observations.")

    model = MaskablePPO.load(model_path, env=env)

    total_rewards = []
    completion_rates = []
    episode_lengths = []
    throughputs = []
    idle_times = []
    validity_rates = []
    rewards = []
    idle_machines_list = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = [False]
        ep_reward = 0.0
        ep_steps = 0
        last_info = {}
        idle_machine_sum = 0.0

        while not done[0]:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, infos = env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1
            if infos and isinstance(infos[0], dict):
                last_info = infos[0]
                idle_machine_sum += float(infos[0].get("idle_machines", 0))

        total_rewards.append(ep_reward)
        rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        completion_rates.append(float(np.clip(last_info.get("completion_rate", 0.0), 0.0, 1.0)))
        jobs_completed = float(last_info.get("jobs_completed", 0))
        throughput = jobs_completed / max(ep_steps, 1)
        avg_idle = idle_machine_sum / max(ep_steps, 1)
        throughputs.append(throughput)
        idle_times.append(avg_idle)
        idle_machines_list.append(avg_idle)
        validity_rates.append(float(np.clip(last_info.get("action_validity_rate", 0.0), 0.0, 1.0)))

    metrics = {
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "avg_completion_rate": float(np.mean(completion_rates)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_throughput": float(np.mean(throughputs)),
        "avg_idle_machines": float(np.mean(idle_times)),
        "avg_action_validity_rate": float(np.mean(validity_rates)),
    }
    print(f"PPO Average Reward: {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"PPO Average Completion Rate: {metrics['avg_completion_rate']:.2%}")
    print(f"PPO Average Episode Length: {metrics['avg_episode_length']:.1f}")
    print(f"PPO Throughput (jobs/step): {metrics['avg_throughput']:.3f}")
    print(f"PPO Avg Idle Machines: {metrics['avg_idle_machines']:.3f}")
    print(f"PPO Action Validity Rate: {metrics['avg_action_validity_rate']:.2%}")

    per_episode = {
        "rewards": rewards,
        "completion_rates": completion_rates,
        "throughputs": throughputs,
        "idle_machines_list": idle_machines_list,
    }
    plot_paths = _plot_eval_curves(per_episode, label=label, include_timestamp=include_timestamp)
    overlay_path = _plot_overlay_from_json(
        current=per_episode,
        baseline_json=overlay_json,
        label=label,
        include_timestamp=include_timestamp,
    )
    print(f"Saved plots to: {PLOTS_DIR}")
    for k, v in plot_paths.items():
        print(f"- {k}: {v}")
    if overlay_path:
        print(f"- reward_overlay: {overlay_path}")
    return metrics


def evaluate_random(task_id: str = "easy", num_episodes: int = 10):
    gym_env_module.DEBUG = False
    print(f"\nEvaluating Random agent | task={task_id} | episodes={num_episodes}")
    env = SmartFactoryEnv(task_id=task_id, render_mode=None)

    total_rewards = []
    completion_rates = []
    throughputs = []
    idle_times = []
    validity_rates = []
    for _ in range(num_episodes):
        _, _ = env.reset()
        done = False
        ep_reward = 0.0
        completion = 0.0
        jobs_completed = 0.0
        ep_steps = 0
        idle_machine_sum = 0.0
        valid_count = 0
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_steps += 1
            completion = float(np.clip(info.get("completion_rate", completion), 0.0, 1.0))
            jobs_completed = float(info.get("jobs_completed", jobs_completed))
            idle_machine_sum += float(info.get("idle_machines", 0))
            valid_count += int(bool(info.get("action_valid", False)))
        total_rewards.append(ep_reward)
        completion_rates.append(completion)
        throughputs.append(jobs_completed / max(ep_steps, 1))
        idle_times.append(idle_machine_sum / max(ep_steps, 1))
        validity_rates.append(valid_count / max(ep_steps, 1))

    metrics = {
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "avg_completion_rate": float(np.mean(completion_rates)),
        "avg_throughput": float(np.mean(throughputs)),
        "avg_idle_machines": float(np.mean(idle_times)),
        "avg_action_validity_rate": float(np.mean(validity_rates)),
    }
    print(f"Random Average Reward: {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"Random Average Completion Rate: {metrics['avg_completion_rate']:.2%}")
    print(f"Random Throughput (jobs/step): {metrics['avg_throughput']:.3f}")
    print(f"Random Avg Idle Machines: {metrics['avg_idle_machines']:.3f}")
    print(f"Random Action Validity Rate: {metrics['avg_action_validity_rate']:.2%}")
    return metrics


def _default_model_paths():
    best_model = os.path.join(_PROJECT_ROOT, "logs", "best_model", "best_model.zip")
    latest_model = os.path.join(MODELS_DIR, "latest.zip")
    latest_stats = os.path.join(MODELS_DIR, "latest_vecnormalize.pkl")
    return best_model, latest_model, latest_stats


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO and random baselines")
    parser.add_argument("--task-id", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--label", type=str, default="experiment", help="Label prefix for saved plots.")
    parser.add_argument(
        "--timestamp-plots",
        action="store_true",
        help="Append timestamp to plot filenames.",
    )
    parser.add_argument(
        "--overlay-json",
        type=str,
        default=None,
        help="Optional path to previous eval_results_*.json for A/B overlay plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    best_model, latest_model, latest_stats = _default_model_paths()
    model_path = args.model_path or (best_model if os.path.exists(best_model) else latest_model)
    stats_path = args.stats_path or latest_stats

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ppo_metrics = evaluate_agent(
        model_path=model_path,
        task_id=args.task_id,
        num_episodes=args.episodes,
        stats_path=stats_path,
        label=args.label,
        include_timestamp=args.timestamp_plots,
        overlay_json=args.overlay_json,
    )
    random_metrics = evaluate_random(task_id=args.task_id, num_episodes=args.episodes)
    print(
        f"\nDelta reward (PPO - Random): {ppo_metrics['avg_reward'] - random_metrics['avg_reward']:.3f}"
    )

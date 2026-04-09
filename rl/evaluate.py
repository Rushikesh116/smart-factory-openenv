from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.baselines import HeuristicBaselinePolicy
from rl.gym_env import SmartFactoryEnv
import rl.gym_env as gym_env_module

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_RL_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")


def _make_eval_vec_env(task_id: str):
    env = DummyVecEnv([lambda: SmartFactoryEnv(task_id=task_id, render_mode=None)])
    return VecMonitor(env)


def _ensure_plots_dir(output_dir: str | None = None) -> str:
    target_dir = output_dir or PLOTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


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


def _timestamp_suffix(label: str, include_timestamp: bool) -> str:
    suffix = f"_{label}" if label else ""
    if include_timestamp:
        suffix = f"{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return suffix


def _plot_eval_curves(
    per_episode: dict[str, list[float]],
    label: str,
    include_timestamp: bool,
    output_dir: str | None = None,
) -> dict[str, str]:
    plots_dir = _ensure_plots_dir(output_dir)
    suffix = _timestamp_suffix(label, include_timestamp)
    paths = {
        "reward_curve": os.path.join(plots_dir, f"reward_curve{suffix}.png"),
        "energy_curve": os.path.join(plots_dir, f"energy_curve{suffix}.png"),
        "throughput_curve": os.path.join(plots_dir, f"throughput_curve{suffix}.png"),
        "delay_curve": os.path.join(plots_dir, f"delay_curve{suffix}.png"),
        "results_json": os.path.join(plots_dir, f"eval_results{suffix}.json"),
    }

    _save_metric_plot(per_episode["rewards"], "Reward per Episode", "Reward", paths["reward_curve"])
    _save_metric_plot(per_episode["avg_energy_usage"], "Energy per Episode", "Energy Usage", paths["energy_curve"])
    _save_metric_plot(per_episode["throughputs"], "Throughput per Episode", "Jobs per Step", paths["throughput_curve"])
    _save_metric_plot(per_episode["avg_delay"], "Delay per Episode", "Average Delay", paths["delay_curve"])

    with open(paths["results_json"], "w", encoding="utf-8") as file:
        json.dump(per_episode, file, indent=2)
    return paths


def _plot_comparison(
    summary: Dict[str, Dict[str, float]],
    label: str,
    include_timestamp: bool,
    output_dir: str | None = None,
) -> str:
    plots_dir = _ensure_plots_dir(output_dir)
    suffix = _timestamp_suffix(label, include_timestamp)
    path = os.path.join(plots_dir, f"comparison_scorecard{suffix}.png")
    policies = list(summary.keys())
    metrics_to_plot = [
        ("avg_reward", "Reward"),
        ("avg_throughput", "Throughput"),
        ("avg_energy_usage", "Energy"),
        ("avg_delay", "Delay"),
    ]
    x = np.arange(len(policies))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for axis, (metric_key, title) in zip(axes, metrics_to_plot):
        values = [summary[policy][metric_key] for policy in policies]
        axis.bar(x, values, color=colors[: len(policies)], alpha=0.9)
        axis.set_title(title)
        axis.set_xticks(x, policies)
        axis.grid(True, axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            axis.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Smart Factory Policy Scorecard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _episode_metric_template() -> Dict[str, list[float]]:
    return {
        "rewards": [],
        "completion_rates": [],
        "episode_lengths": [],
        "throughputs": [],
        "avg_energy_usage": [],
        "energy_over_budget_rate": [],
        "avg_delay": [],
        "p95_delay": [],
        "max_delay": [],
        "avg_breakdown_risk": [],
        "breakdown_rate": [],
        "action_validity_rate": [],
    }


def _summarize_episode_metrics(per_episode: Dict[str, list[float]]) -> Dict[str, float]:
    return {
        "avg_reward": float(np.mean(per_episode["rewards"])),
        "std_reward": float(np.std(per_episode["rewards"])),
        "avg_completion_rate": float(np.mean(per_episode["completion_rates"])),
        "avg_episode_length": float(np.mean(per_episode["episode_lengths"])),
        "avg_throughput": float(np.mean(per_episode["throughputs"])),
        "avg_energy_usage": float(np.mean(per_episode["avg_energy_usage"])),
        "avg_delay": float(np.mean(per_episode["avg_delay"])),
        "p95_delay": float(np.mean(per_episode["p95_delay"])),
        "max_delay": float(np.mean(per_episode["max_delay"])),
        "avg_breakdown_risk": float(np.mean(per_episode["avg_breakdown_risk"])),
        "avg_breakdown_rate": float(np.mean(per_episode["breakdown_rate"])),
        "avg_energy_over_budget_rate": float(np.mean(per_episode["energy_over_budget_rate"])),
        "avg_action_validity_rate": float(np.mean(per_episode["action_validity_rate"])),
    }


def _collect_step_metrics(step_infos: list[Dict[str, Any]]) -> Dict[str, float]:
    energies = [float(info.get("current_energy_usage", 0.0)) for info in step_infos]
    delays = [float(info.get("avg_waiting_time", info.get("delay", 0.0))) for info in step_infos]
    breakdowns = [float(info.get("breakdown_risk", 0.0)) for info in step_infos]
    breakdown_events = float(step_infos[-1].get("breakdown_events", 0.0)) if step_infos else 0.0
    energy_budget = float(step_infos[-1].get("energy_budget", 1.0)) if step_infos else 1.0
    if energy_budget <= 0:
        energy_budget = 1.0

    over_budget_steps = sum(1 for value in energies if value > energy_budget)
    return {
        "avg_energy_usage": float(np.mean(energies)) if energies else 0.0,
        "energy_over_budget_rate": over_budget_steps / max(len(energies), 1),
        "avg_delay": float(np.mean(delays)) if delays else 0.0,
        "p95_delay": float(np.percentile(delays, 95)) if delays else 0.0,
        "max_delay": float(np.max(delays)) if delays else 0.0,
        "avg_breakdown_risk": float(np.mean(breakdowns)) if breakdowns else 0.0,
        "breakdown_rate": breakdown_events / max(len(step_infos), 1),
    }


def evaluate_agent(
    model_path: str,
    task_id: str = "easy",
    num_episodes: int = 10,
    stats_path: str | None = None,
    label: str = "ppo",
    include_timestamp: bool = False,
    output_dir: str | None = None,
) -> tuple[Dict[str, float], Dict[str, list[float]], Dict[str, str]]:
    gym_env_module.DEBUG = False
    env = _make_eval_vec_env(task_id)

    if stats_path and os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    model = MaskablePPO.load(model_path, env=env)
    per_episode = _episode_metric_template()

    for _ in range(num_episodes):
        obs = env.reset()
        done = [False]
        episode_reward = 0.0
        step_infos: list[Dict[str, Any]] = []
        step_count = 0

        while not done[0]:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, infos = env.step(action)
            info = infos[0] if infos and isinstance(infos[0], dict) else {}
            step_infos.append(info)
            episode_reward += float(reward[0])
            step_count += 1

        step_metrics = _collect_step_metrics(step_infos)
        last_info = step_infos[-1] if step_infos else {}
        per_episode["rewards"].append(episode_reward)
        per_episode["completion_rates"].append(float(last_info.get("completion_rate", 0.0)))
        per_episode["episode_lengths"].append(step_count)
        per_episode["throughputs"].append(float(last_info.get("throughput", 0.0)))
        per_episode["action_validity_rate"].append(float(last_info.get("action_validity_rate", 0.0)))
        for key, value in step_metrics.items():
            per_episode[key].append(float(value))

    metrics = _summarize_episode_metrics(per_episode)
    plot_paths = _plot_eval_curves(
        per_episode,
        label=label,
        include_timestamp=include_timestamp,
        output_dir=output_dir,
    )
    return metrics, per_episode, plot_paths


def _run_policy(
    env: SmartFactoryEnv,
    num_episodes: int,
    action_fn: Callable[[Any, Any], int],
    label: str,
    include_timestamp: bool,
    output_dir: str | None = None,
) -> tuple[Dict[str, float], Dict[str, list[float]], Dict[str, str]]:
    per_episode = _episode_metric_template()

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_infos: list[Dict[str, Any]] = []
        step_count = 0

        while not done:
            action = int(action_fn(env, obs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            step_infos.append(info)
            step_count += 1

        step_metrics = _collect_step_metrics(step_infos)
        last_info = step_infos[-1] if step_infos else {}
        per_episode["rewards"].append(episode_reward)
        per_episode["completion_rates"].append(float(last_info.get("completion_rate", 0.0)))
        per_episode["episode_lengths"].append(step_count)
        per_episode["throughputs"].append(float(last_info.get("throughput", 0.0)))
        per_episode["action_validity_rate"].append(float(last_info.get("action_validity_rate", 0.0)))
        for key, value in step_metrics.items():
            per_episode[key].append(float(value))

    metrics = _summarize_episode_metrics(per_episode)
    plot_paths = _plot_eval_curves(
        per_episode,
        label=label,
        include_timestamp=include_timestamp,
        output_dir=output_dir,
    )
    return metrics, per_episode, plot_paths


def evaluate_random(
    task_id: str = "easy",
    num_episodes: int = 10,
    label: str = "random",
    include_timestamp: bool = False,
    output_dir: str | None = None,
) -> tuple[Dict[str, float], Dict[str, list[float]], Dict[str, str]]:
    gym_env_module.DEBUG = False
    env = SmartFactoryEnv(task_id=task_id, render_mode=None)
    return _run_policy(
        env,
        num_episodes,
        lambda current_env, _obs: current_env.action_space.sample(),
        label,
        include_timestamp,
        output_dir,
    )


def evaluate_baseline(
    task_id: str = "easy",
    num_episodes: int = 10,
    label: str = "baseline",
    include_timestamp: bool = False,
    output_dir: str | None = None,
) -> tuple[Dict[str, float], Dict[str, list[float]], Dict[str, str]]:
    gym_env_module.DEBUG = False
    env = SmartFactoryEnv(task_id=task_id, render_mode=None)
    policy = HeuristicBaselinePolicy()
    return _run_policy(env, num_episodes, policy.act, label, include_timestamp, output_dir)


def _default_model_paths() -> tuple[str, str, str]:
    best_model = os.path.join(_PROJECT_ROOT, "logs", "best_model", "best_model.zip")
    latest_model = os.path.join(MODELS_DIR, "latest.zip")
    latest_stats = os.path.join(MODELS_DIR, "latest_vecnormalize.pkl")
    return best_model, latest_model, latest_stats


def _save_comparison_summary(
    summary: Dict[str, Dict[str, float]],
    label: str,
    include_timestamp: bool,
    output_dir: str | None = None,
) -> Dict[str, str]:
    plots_dir = _ensure_plots_dir(output_dir)
    suffix = _timestamp_suffix(label, include_timestamp)
    json_path = os.path.join(plots_dir, f"comparison_summary{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    csv_path = os.path.join(plots_dir, f"comparison_summary{suffix}.csv")
    fieldnames = ["policy", *next(iter(summary.values())).keys()]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name, metrics in summary.items():
            writer.writerow({"policy": policy_name, **metrics})
    markdown_path = os.path.join(plots_dir, f"comparison_summary{suffix}.md")
    with open(markdown_path, "w", encoding="utf-8") as file:
        headers = ["Policy", "Reward", "Energy", "Throughput", "Delay", "Breakdown Risk", "Completion"]
        rows = []
        for policy_name, metrics in summary.items():
            rows.append(
                [
                    policy_name,
                    f"{metrics['avg_reward']:.3f}",
                    f"{metrics['avg_energy_usage']:.3f}",
                    f"{metrics['avg_throughput']:.3f}",
                    f"{metrics['avg_delay']:.3f}",
                    f"{metrics['avg_breakdown_risk']:.3f}",
                    f"{metrics['avg_completion_rate']:.3f}",
                ]
            )
        file.write("# Smart Factory Evaluation Summary\n\n")
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            file.write("| " + " | ".join(row) + " |\n")
    log_path = os.path.join(plots_dir, f"comparison_log{suffix}.txt")
    with open(log_path, "w", encoding="utf-8") as file:
        for policy_name, metrics in summary.items():
            file.write(
                f"{policy_name}: reward={metrics['avg_reward']:.3f}, energy={metrics['avg_energy_usage']:.3f}, "
                f"throughput={metrics['avg_throughput']:.3f}, delay={metrics['avg_delay']:.3f}, "
                f"breakdown_risk={metrics['avg_breakdown_risk']:.3f}\n"
            )
    return {
        "comparison_json": json_path,
        "comparison_csv": csv_path,
        "comparison_markdown": markdown_path,
        "comparison_log": log_path,
        "comparison_plot": _plot_comparison(summary, label, include_timestamp, output_dir),
    }


def evaluate_policies(
    model_path: str,
    task_id: str = "easy",
    num_episodes: int = 10,
    stats_path: str | None = None,
    label: str = "comparison",
    include_timestamp: bool = False,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    ppo_metrics, _, ppo_paths = evaluate_agent(
        model_path=model_path,
        task_id=task_id,
        num_episodes=num_episodes,
        stats_path=stats_path,
        label="ppo",
        include_timestamp=include_timestamp,
        output_dir=output_dir,
    )
    baseline_metrics, _, baseline_paths = evaluate_baseline(
        task_id=task_id,
        num_episodes=num_episodes,
        label="baseline",
        include_timestamp=include_timestamp,
        output_dir=output_dir,
    )
    random_metrics, _, random_paths = evaluate_random(
        task_id=task_id,
        num_episodes=num_episodes,
        label="random",
        include_timestamp=include_timestamp,
        output_dir=output_dir,
    )

    summary = {
        "ppo": ppo_metrics,
        "baseline": baseline_metrics,
        "random": random_metrics,
    }
    summary_paths = _save_comparison_summary(summary, label, include_timestamp, output_dir)
    return {
        "summary": summary,
        "artifacts": {
            "ppo": ppo_paths,
            "baseline": baseline_paths,
            "random": random_paths,
            "comparison": summary_paths,
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO, heuristic baseline, and random policies")
    parser.add_argument("--task-id", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--label", type=str, default="comparison")
    parser.add_argument("--timestamp-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    best_model, latest_model, latest_stats = _default_model_paths()
    model_path = args.model_path or (best_model if os.path.exists(best_model) else latest_model)
    stats_path = args.stats_path or latest_stats

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ppo_metrics, _, ppo_paths = evaluate_agent(
        model_path=model_path,
        task_id=args.task_id,
        num_episodes=args.episodes,
        stats_path=stats_path,
        label="ppo",
        include_timestamp=args.timestamp_plots,
    )
    baseline_metrics, _, baseline_paths = evaluate_baseline(
        task_id=args.task_id,
        num_episodes=args.episodes,
        label="baseline",
        include_timestamp=args.timestamp_plots,
    )
    random_metrics, _, random_paths = evaluate_random(
        task_id=args.task_id,
        num_episodes=args.episodes,
        label="random",
        include_timestamp=args.timestamp_plots,
    )

    summary = {
        "ppo": ppo_metrics,
        "baseline": baseline_metrics,
        "random": random_metrics,
    }
    summary_paths = _save_comparison_summary(summary, args.label, args.timestamp_plots)

    print("\nEvaluation Summary")
    for policy_name, metrics in summary.items():
        print(
            f"- {policy_name}: reward={metrics['avg_reward']:.3f}, "
            f"energy={metrics['avg_energy_usage']:.3f}, throughput={metrics['avg_throughput']:.3f}, "
            f"delay={metrics['avg_delay']:.3f}, breakdown_risk={metrics['avg_breakdown_risk']:.3f}"
        )

    print("\nSaved artifacts:")
    for group in (ppo_paths, baseline_paths, random_paths, summary_paths):
        for key, value in group.items():
            print(f"- {key}: {value}")

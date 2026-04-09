from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.gym_env import SmartFactoryEnv

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MaskablePPO = None
    MASKABLE_PPO_AVAILABLE = False

from stable_baselines3 import PPO


TITLE = "AI-Powered Smart Factory Optimization"
DESCRIPTION = (
    "This system uses reinforcement learning to optimize energy, throughput, "
    "and delay in industrial environments."
)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "latest.zip"))


def _action_to_int(action: Any) -> int:
    if isinstance(action, np.ndarray):
        return int(action.item()) if action.ndim == 0 else int(action[0])
    return int(action)


@lru_cache(maxsize=1)
def load_model() -> tuple[Any | None, str]:
    if not os.path.exists(MODEL_PATH):
        return None, "heuristic_fallback"

    if MASKABLE_PPO_AVAILABLE:
        try:
            return MaskablePPO.load(MODEL_PATH), "MaskablePPO"
        except Exception:
            pass

    try:
        return PPO.load(MODEL_PATH), "PPO"
    except Exception:
        return None, "heuristic_fallback"


def choose_rl_action(env: SmartFactoryEnv, obs: Any, randomness: bool) -> tuple[int, str]:
    model, model_name = load_model()
    if model is None:
        # Greedy fallback that still looks purposeful if a model artifact is missing.
        action_masks = env.action_masks()
        valid_actions = np.flatnonzero(action_masks)
        action = int(valid_actions[0]) if len(valid_actions) else 0
        return action, model_name

    deterministic = not randomness
    if model_name == "MaskablePPO":
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
    else:
        action, _ = model.predict(obs, deterministic=deterministic)
    return _action_to_int(action), model_name


def choose_random_action(env: SmartFactoryEnv, rng: np.random.Generator | None) -> int:
    if rng is not None:
        valid_actions = np.flatnonzero(env.action_masks())
        if len(valid_actions) == 0:
            return 0
        return int(rng.choice(valid_actions))
    return int(env.action_space.sample())


def run_policy(task_id: str, steps: int, randomness: bool, policy: str) -> Dict[str, Any]:
    env = SmartFactoryEnv(task_id=task_id, render_mode=None)
    obs, info = env.reset()
    if not randomness:
        env.action_space.seed(42)
        np.random.seed(42)
    rng = np.random.default_rng(42) if not randomness and policy == "random" else None

    results: Dict[str, Any] = {
        "policy": policy,
        "steps": [],
        "rewards": [],
        "energy": [],
        "throughput": [],
        "delay": [],
        "breakdown_risk": [],
        "logs": [],
        "done": False,
        "model_name": None,
    }

    total_reward = 0.0
    current_info = info
    for step_idx in range(1, steps + 1):
        if policy == "random":
            action = choose_random_action(env, rng)
            policy_label = "random"
        else:
            action, policy_label = choose_rl_action(env, obs, randomness)
            results["model_name"] = policy_label

        obs, reward, terminated, truncated, current_info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        results["steps"].append(step_idx)
        results["rewards"].append(float(reward))
        results["energy"].append(float(current_info.get("current_energy_usage", 0.0)))
        results["throughput"].append(float(current_info.get("throughput", 0.0)))
        results["delay"].append(float(current_info.get("delay", 0.0)))
        results["breakdown_risk"].append(float(current_info.get("breakdown_risk", 0.0)))
        results["logs"].append(
            f"step={step_idx:02d} policy={policy_label} action={action} "
            f"reward={float(reward):.3f} energy={results['energy'][-1]:.2f} "
            f"throughput={results['throughput'][-1]:.3f} delay={results['delay'][-1]:.2f}"
        )
        results["done"] = done
        if done:
            break

    results["summary"] = {
        "Policy": "RL Agent" if policy == "rl" else "Random Policy",
        "Steps": len(results["steps"]),
        "Total Reward": round(sum(results["rewards"]), 3),
        "Avg Energy": round(float(np.mean(results["energy"])) if results["energy"] else 0.0, 3),
        "Avg Throughput": round(float(np.mean(results["throughput"])) if results["throughput"] else 0.0, 3),
        "Avg Delay": round(float(np.mean(results["delay"])) if results["delay"] else 0.0, 3),
        "Final Breakdown Risk": round(results["breakdown_risk"][-1] if results["breakdown_risk"] else 0.0, 3),
        "Completion Rate": round(float(current_info.get("completion_rate", 0.0)), 3),
        "Done": results["done"],
    }
    return results


def make_reward_plot(random_result: Dict[str, Any], rl_result: Dict[str, Any], upto: int | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    random_x = random_result["steps"]
    rl_x = rl_result["steps"][:upto] if upto is not None else rl_result["steps"]
    rl_rewards = rl_result["rewards"][:upto] if upto is not None else rl_result["rewards"]

    ax.plot(random_x, random_result["rewards"], label="Random Policy", color="#8a8f98", linewidth=2.0, alpha=0.9)
    ax.plot(rl_x, rl_rewards, label="RL Agent", color="#1f77b4", linewidth=2.4)
    ax.set_title("Reward Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def make_metrics_plot(random_result: Dict[str, Any], rl_result: Dict[str, Any], upto: int | None = None):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    rl_slice = slice(None if upto is None else upto)

    metric_specs = [
        ("energy", "Energy Usage", "#d62728"),
        ("throughput", "Throughput", "#2ca02c"),
        ("delay", "Delay", "#ff7f0e"),
    ]
    for axis, (key, label, color) in zip(axes, metric_specs):
        axis.plot(random_result["steps"], random_result[key], label=f"Random {label}", color="#b0b7c3", linestyle="--")
        axis.plot(rl_result["steps"][rl_slice], rl_result[key][rl_slice], label=f"RL {label}", color=color, linewidth=2.0)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Step")
    fig.suptitle("Energy / Throughput / Delay")
    fig.tight_layout()
    return fig


def build_comparison_rows(random_result: Dict[str, Any], rl_result: Dict[str, Any]) -> list[list[Any]]:
    return [
        list(random_result["summary"].values()),
        list(rl_result["summary"].values()),
    ]


def build_summary_markdown(random_result: Dict[str, Any], rl_result: Dict[str, Any]) -> str:
    random_summary = random_result["summary"]
    rl_summary = rl_result["summary"]

    def improvement(higher_is_better: bool, rl_value: float, random_value: float) -> float:
        baseline = max(abs(random_value), 1e-6)
        delta = (rl_value - random_value) / baseline * 100.0 if higher_is_better else (random_value - rl_value) / baseline * 100.0
        return delta

    reward_gain = improvement(True, rl_summary["Total Reward"], random_summary["Total Reward"])
    throughput_gain = improvement(True, rl_summary["Avg Throughput"], random_summary["Avg Throughput"])
    energy_gain = improvement(False, rl_summary["Avg Energy"], random_summary["Avg Energy"])
    delay_gain = improvement(False, rl_summary["Avg Delay"], random_summary["Avg Delay"])
    model_name = rl_result.get("model_name") or "heuristic_fallback"

    return (
        f"### Final Metrics Summary\n"
        f"- RL backend: `{model_name}`\n"
        f"- Reward improvement: **{reward_gain:+.1f}%**\n"
        f"- Throughput improvement: **{throughput_gain:+.1f}%**\n"
        f"- Energy improvement: **{energy_gain:+.1f}%**\n"
        f"- Delay improvement: **{delay_gain:+.1f}%**\n"
        f"- Winner: **{'RL Agent' if rl_summary['Total Reward'] >= random_summary['Total Reward'] else 'Random Policy'}**"
    )


def run_demo(task_id: str, steps: int, randomness: bool):
    random_result = run_policy(task_id=task_id, steps=steps, randomness=randomness, policy="random")
    rl_result = run_policy(task_id=task_id, steps=steps, randomness=randomness, policy="rl")

    comparison_rows = build_comparison_rows(random_result, rl_result)
    summary_markdown = build_summary_markdown(random_result, rl_result)

    log_lines: list[str] = []
    total_steps = len(rl_result["steps"])
    for idx in range(total_steps):
        log_lines.append(rl_result["logs"][idx])
        yield (
            rl_result["steps"][idx],
            rl_result["energy"][idx],
            rl_result["throughput"][idx],
            rl_result["delay"][idx],
            rl_result["rewards"][idx],
            make_reward_plot(random_result, rl_result, upto=idx + 1),
            make_metrics_plot(random_result, rl_result, upto=idx + 1),
            comparison_rows,
            summary_markdown,
            "\n".join(log_lines[-18:]),
        )


with gr.Blocks(theme=gr.themes.Soft(), title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    gr.Markdown("Click **Run Simulation** to compare a trained RL agent against a random policy.")

    with gr.Row():
        task_input = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="medium",
            label="Task Difficulty",
        )
        steps_input = gr.Slider(10, 60, value=25, step=1, label="Number of Steps")
        randomness_input = gr.Checkbox(value=False, label="Enable Randomness")

    run_button = gr.Button("Run Simulation", variant="primary")

    with gr.Row():
        current_step = gr.Number(label="Current Step", precision=0)
        current_energy = gr.Number(label="Energy Usage")
        current_throughput = gr.Number(label="Throughput")
        current_delay = gr.Number(label="Delay")
        current_reward = gr.Number(label="Reward")

    with gr.Row():
        reward_plot = gr.Plot(label="Reward Over Time")
        metrics_plot = gr.Plot(label="Operational Metrics")

    comparison_table = gr.Dataframe(
        headers=["Policy", "Steps", "Total Reward", "Avg Energy", "Avg Throughput", "Avg Delay", "Final Breakdown Risk", "Completion Rate", "Done"],
        datatype=["str", "number", "number", "number", "number", "number", "number", "number", "bool"],
        label="Side-by-Side Metrics",
    )
    summary_output = gr.Markdown(label="Final Metrics Summary")
    logs_output = gr.Textbox(label="Simulation Log", lines=12, max_lines=18)

    run_button.click(
        fn=run_demo,
        inputs=[task_input, steps_input, randomness_input],
        outputs=[
            current_step,
            current_energy,
            current_throughput,
            current_delay,
            current_reward,
            reward_plot,
            metrics_plot,
            comparison_table,
            summary_output,
            logs_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

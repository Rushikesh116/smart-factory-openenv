import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, title="Training Rewards", save_path="training_rewards.png"):
    """
    Plots the training rewards over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def log_metrics(metrics, log_path="training_metrics.json"):
    """
    Logs training metrics to a JSON file.
    """
    import json
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics logged to {log_path}")

def get_latest_model(directory=".", prefix="ppo_smart_factory"):
    """
    Finds the latest trained model in a directory.
    """
    models = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".zip")]
    if not models:
        return None
    # Sort by modification time
    models.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, models[0])


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_run_name(task_id: str, prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{task_id}_{ts}"


def make_model_paths(models_dir: str, run_name: str) -> dict:
    ensure_dir(models_dir)
    return {
        "run_model": os.path.join(models_dir, f"{run_name}.zip"),
        "run_stats": os.path.join(models_dir, f"{run_name}_vecnormalize.pkl"),
        "latest_model": os.path.join(models_dir, "latest.zip"),
        "latest_stats": os.path.join(models_dir, "latest_vecnormalize.pkl"),
    }


def copy_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        shutil.copy2(src, dst)

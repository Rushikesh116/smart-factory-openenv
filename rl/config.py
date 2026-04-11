from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    pi_layers: tuple[int, int] = (128, 128)
    vf_layers: tuple[int, int] = (128, 128)
    use_linear_lr_schedule: bool = True


@dataclass(frozen=True)
class TrainConfig:
    task_id: str = "medium"
    timesteps: int = 20000
    eval_freq: int = 5000
    checkpoint_freq: int = 5000
    curriculum_easy_timesteps: int = 30000
    curriculum_medium_timesteps: int = 50000
    curriculum_hard_timesteps: int = 100000

"""Default configuration values for experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    # Environment
    episodes: int = 500
    rollout_length: int = 128
    n_players_per_team: int = 2
    grid_height: int = 5
    grid_width: int = 7
    max_steps: int = 200

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    update_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Misc
    log_dir: Path = Path("results/tensorboard")
    checkpoint_dir: Path = Path("models")
    save_interval: int = 50


@dataclass
class EvaluationConfig:
    episodes: int = 10
    render: bool = False

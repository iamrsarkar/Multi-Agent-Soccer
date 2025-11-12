"""Utilities to orchestrate self-play and opponent sampling."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .ppo_agent import ActorConfig, ActorNetwork


@dataclass
class SelfPlayConfig:
    """Configuration governing the self-play opponent pool."""

    snapshot_interval: int = 10
    max_pool_size: int = 5
    random_opponent_prob: float = 0.1


class FrozenPolicy:
    """Non-trainable actor used for opponent sampling."""

    def __init__(self, actor_config: ActorConfig, state_dict: dict, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.actor = ActorNetwork(actor_config).to(self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        tensor_obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(tensor_obs)
        action = torch.argmax(logits, dim=-1)
        return int(action.item())


class RandomPolicy:
    """Fallback opponent that selects actions uniformly at random."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, observation: np.ndarray) -> int:  # noqa: D401 - signature compatibility
        return random.randrange(self.action_dim)


@dataclass
class SelfPlayManager:
    """Maintains a pool of opponent policies for self-play training."""

    actor_config: ActorConfig
    cfg: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self._pool: List[FrozenPolicy] = []
        self._episodes_since_snapshot = 0
        self._random_policy = RandomPolicy(self.actor_config.action_dim)

    def sample_opponent(self) -> object:
        """Return a policy object implementing ``act(observation)``."""

        if not self._pool or random.random() < self.cfg.random_opponent_prob:
            return self._random_policy
        return random.choice(self._pool)

    def maybe_add_snapshot(self, actor_state: dict) -> None:
        """Store a snapshot of the learner into the opponent pool."""

        self._episodes_since_snapshot += 1
        if self._episodes_since_snapshot < self.cfg.snapshot_interval:
            return

        snapshot = FrozenPolicy(self.actor_config, actor_state, device=self.device)
        self._pool.append(snapshot)
        self._episodes_since_snapshot = 0
        if len(self._pool) > self.cfg.max_pool_size:
            self._pool.pop(0)

    def clear(self) -> None:
        self._pool.clear()
        self._episodes_since_snapshot = 0

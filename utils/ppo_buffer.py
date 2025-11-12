"""Rollout buffer supporting Generalised Advantage Estimation for PPO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List

import numpy as np
import torch


@dataclass
class BufferConfig:
    capacity: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: torch.device | str = "cpu"


class RolloutBuffer:
    def __init__(self, cfg: BufferConfig, obs_dim: int, state_dim: int) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.clear()

    def store(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        joint_state: np.ndarray,
    ) -> None:
        self.observations.append(observation.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.joint_states.append(joint_state.copy())

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        returns = []
        advantages = []
        gae = 0.0
        values = self.values + [last_value]
        for step in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[step])
            delta = self.rewards[step] + self.cfg.gamma * values[step + 1] * mask - values[step]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        self.returns = returns
        self.advantages = advantages

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        observations = torch.as_tensor(self.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.int64, device=self.device)
        log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.advantages, dtype=torch.float32, device=self.device)
        joint_states = torch.as_tensor(self.joint_states, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "joint_states": joint_states,
        }

    def mini_batches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        tensors = self.get_tensors()
        total = tensors["observations"].shape[0]
        indices = torch.randperm(total)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_idx = indices[start:end]
            yield {key: tensor[batch_idx] for key, tensor in tensors.items()}

    def clear(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.joint_states: List[np.ndarray] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

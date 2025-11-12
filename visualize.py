"""Visualisation utilities for analysing trained soccer agents."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from agents.ppo_agent import ActorConfig, PPOAgent
from envs.soccer_env import SoccerEnv, SoccerEnvConfig


def run_episode(
    env: SoccerEnv, actor: PPOAgent | None, deterministic: bool = True, render: bool = False
) -> Dict[str, List[Tuple[int, int]]]:
    observations, _ = env.reset()
    possible_agents = env.possible_agents
    trajectories: Dict[str, List[Tuple[int, int]]] = {agent: [] for agent in possible_agents}
    done = {agent: False for agent in possible_agents}

    while not all(done.values()):
        actions = {}
        for agent in possible_agents:
            if agent not in observations:
                continue
            if actor is not None:
                action, _ = actor.select_action(observations[agent], deterministic=deterministic)
            else:
                action = env.action_space(agent).sample()
            actions[agent] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = {agent: terminations[agent] or truncations[agent] for agent in possible_agents}
        for agent in possible_agents:
            if agent in env._positions:  # type: ignore[attr-defined]
                trajectories[agent].append(env._positions[agent])  # type: ignore[attr-defined]
        if render:
            env.render()
    return trajectories


def plot_trajectories(trajectories: Dict[str, List[Tuple[int, int]]], grid_shape: Tuple[int, int]) -> None:
    plt.figure(figsize=(6, 6))
    height, width = grid_shape
    for agent, coords in trajectories.items():
        if not coords:
            continue
        y = [pos[0] for pos in coords]
        x = [pos[1] for pos in coords]
        plt.plot(x, y, marker="o", label=agent)
        plt.text(x[0], y[0], f"{agent} start", fontsize=8)
    plt.xlim(-0.5, width - 0.5)
    plt.ylim(height - 0.5, -0.5)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Field width")
    plt.ylabel("Field height")
    plt.title("Agent trajectories")
    plt.show()


def visualise(model_path: Path | None, cfg: SoccerEnvConfig, render: bool) -> None:
    env = SoccerEnv(cfg)
    actor: PPOAgent | None = None
    if model_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        action_dim = env.action_space(env.possible_agents[0]).n
        actor_cfg = ActorConfig(obs_dim=obs_dim, action_dim=action_dim)
        actor = PPOAgent(actor_cfg, device=device)
        state_dict = torch.load(model_path, map_location=device)
        actor.load_state_dict(state_dict)
    trajectories = run_episode(env, actor, render=render)
    if not render:
        plot_trajectories(trajectories, (cfg.grid_height, cfg.grid_width))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise a match episode")
    parser.add_argument("--model", type=Path, default=None, help="Optional path to an actor checkpoint")
    parser.add_argument("--render", action="store_true", help="Render the episode in real-time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SoccerEnvConfig(n_players_per_team=11)
    model_path = args.model if args.model is not None and args.model != Path("None") else None
    visualise(model_path, cfg, args.render)


if __name__ == "__main__":
    main()

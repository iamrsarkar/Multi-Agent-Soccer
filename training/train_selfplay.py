"""Training loop for self-play PPO with a centralized critic."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.centralized_critic import CentralizedCritic, CriticConfig
from agents.ppo_agent import ActorConfig, PPOAgent
from agents.selfplay_manager import SelfPlayConfig, SelfPlayManager
from configs.defaults import TrainingConfig
from envs.soccer_env import SoccerEnv, SoccerEnvConfig
from utils.ppo_buffer import BufferConfig, RolloutBuffer


def build_joint_state(
    observations: Dict[str, np.ndarray], agent_order: List[str], obs_dim: int
) -> np.ndarray:
    """Concatenate observations into a joint state vector."""

    features: List[np.ndarray] = []
    zero = np.zeros(obs_dim, dtype=np.float32)
    for agent in agent_order:
        features.append(observations.get(agent, zero))
    return np.concatenate(features, axis=0)


def train(cfg: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = SoccerEnvConfig(
        grid_height=cfg.grid_height,
        grid_width=cfg.grid_width,
        n_players_per_team=cfg.n_players_per_team,
        max_steps=cfg.max_steps,
    )
    env = SoccerEnv(env_cfg)

    possible_agents = env.possible_agents
    learning_team = "red"
    learner_ids = [agent for agent in possible_agents if agent.startswith(learning_team)]
    obs_dim = env.observation_space(possible_agents[0]).shape[0]
    action_dim = env.action_space(possible_agents[0]).n

    actor_cfg = ActorConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=128,
        hidden_layers=2,
        lr=cfg.learning_rate,
        clip_param=cfg.clip_param,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
    )
    critic_cfg = CriticConfig(input_dim=obs_dim * len(possible_agents), hidden_dim=256, hidden_layers=2)

    learner = PPOAgent(actor_cfg, device=device)
    critic = CentralizedCritic(critic_cfg).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)
    buffer_cfg = BufferConfig(
        capacity=cfg.rollout_length * len(learner_ids),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        device=device,
    )
    buffer = RolloutBuffer(buffer_cfg, obs_dim=obs_dim, state_dim=critic_cfg.input_dim)
    selfplay_manager = SelfPlayManager(actor_cfg, SelfPlayConfig(), device=device)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(cfg.log_dir) / timestamp
    writer = SummaryWriter(log_dir=str(log_dir))

    global_step = 0
    for episode in range(1, cfg.episodes + 1):
        observations, _ = env.reset()
        buffer.clear()
        opponent_policy = selfplay_manager.sample_opponent()
        episode_reward = 0.0
        episode_steps = 0
        done_flags = {agent: False for agent in possible_agents}

        for t in range(cfg.rollout_length):
            joint_state = build_joint_state(observations, possible_agents, obs_dim)
            state_tensor = torch.as_tensor(joint_state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = float(critic(state_tensor).item())

            actions: Dict[str, int] = {}
            log_probs: Dict[str, float] = {}
            for agent_id in possible_agents:
                if agent_id not in observations:
                    continue
                if agent_id.startswith(learning_team):
                    action, log_prob = learner.select_action(observations[agent_id])
                    actions[agent_id] = action
                    log_probs[agent_id] = log_prob
                else:
                    actions[agent_id] = opponent_policy.act(observations[agent_id])

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done_flags = {agent: terminations[agent] or truncations[agent] for agent in possible_agents}

            for agent_id in learner_ids:
                if agent_id in observations:
                    buffer.store(
                        observation=observations[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs[agent_id],
                        reward=rewards.get(agent_id, 0.0),
                        value=value,
                        done=done_flags.get(agent_id, False),
                        joint_state=joint_state,
                    )
                    episode_reward += rewards.get(agent_id, 0.0)

            episode_steps += 1

            observations = next_obs
            global_step += 1

            if all(done_flags.values()):
                break

        final_state = build_joint_state(observations, possible_agents, obs_dim)
        final_tensor = torch.as_tensor(final_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(critic(final_tensor).item())
        buffer.compute_returns_and_advantages(last_value)

        # PPO update
        for _ in range(cfg.update_epochs):
            for batch in buffer.mini_batches(cfg.batch_size):
                new_log_probs, entropy = learner.evaluate_actions(batch["observations"], batch["actions"])
                ratio = torch.exp(new_log_probs - batch["log_probs"])
                surrogate1 = ratio * batch["advantages"]
                surrogate2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * batch["advantages"]
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                entropy_loss = entropy.mean()

                critic_values = critic(batch["joint_states"])
                value_loss = (batch["returns"] - critic_values).pow(2).mean()

                actor_loss = policy_loss - cfg.entropy_coef * entropy_loss
                learner.update(actor_loss)

                critic_optimizer.zero_grad()
                (cfg.value_loss_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                critic_optimizer.step()

        avg_reward = episode_reward / max(len(learner_ids), 1)
        writer.add_scalar("train/episode_reward", avg_reward, episode)
        writer.add_scalar("train/episode_length", episode_steps, episode)

        if episode % cfg.save_interval == 0:
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(learner.state_dict(), cfg.checkpoint_dir / f"soccer_actor_ep{episode}.pth")
            torch.save(critic.state_dict(), cfg.checkpoint_dir / f"soccer_critic_ep{episode}.pth")

        selfplay_manager.maybe_add_snapshot(learner.state_dict())

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{cfg.episodes}: reward={avg_reward:.3f}, steps={episode_steps}"
            )

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play PPO training entry point")
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--rollout-length", type=int, default=TrainingConfig.rollout_length)
    parser.add_argument("--log-dir", type=Path, default=TrainingConfig.log_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=TrainingConfig.checkpoint_dir)
    parser.add_argument("--save-interval", type=int, default=TrainingConfig.save_interval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        episodes=args.episodes,
        rollout_length=args.rollout_length,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
    )
    train(cfg)


if __name__ == "__main__":
    main()

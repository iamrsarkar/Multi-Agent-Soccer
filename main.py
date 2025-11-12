"""Command line entry point for training Multi-Agent Soccer policies."""
from __future__ import annotations

import argparse

from configs.defaults import TrainingConfig
from training.train_selfplay import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Soccer training launcher")
    parser.add_argument("--env", type=str, default="soccer", help="Environment identifier (currently only soccer)")
    parser.add_argument("--algo", type=str, default="selfplay_ppo", help="Training algorithm")
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--rollout-length", type=int, default=TrainingConfig.rollout_length)
    parser.add_argument("--save-interval", type=int, default=TrainingConfig.save_interval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env != "soccer":
        raise ValueError("Only the soccer environment is implemented in this template.")
    if args.algo != "selfplay_ppo":
        raise ValueError("Only selfplay_ppo is available in this version.")

    cfg = TrainingConfig(
        episodes=args.episodes,
        rollout_length=args.rollout_length,
        save_interval=args.save_interval,
    )
    train(cfg)


if __name__ == "__main__":
    main()

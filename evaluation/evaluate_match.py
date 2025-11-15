import argparse
import torch
from envs.soccer_env_3v3 import SoccerEnv3v3
from agents.ppo_agent import PPOAgent

def evaluate(args):
    """
    Runs a continuous evaluation of a trained model in the soccer environment.
    """
    env = SoccerEnv3v3(render_mode="human")

    # Get observation and action space sizes from the environment
    obs_space_size = env.observation_space["player_0"].shape[0]
    action_space_size = env.action_space["player_0"].n

    # Initialize the PPO agent
    model = PPOAgent(obs_space_size, action_space_size)

    # Load the trained model weights
    model.load_state_dict(torch.load(args.model))
    model.eval()

    try:
        while True:
            observations, _ = env.reset()
            terminated = False

            while not terminated:
                env.render()

                actions = {}
                for agent_id, obs in observations.items():
                    # Convert observation to a tensor
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                    # Get action from the model
                    with torch.no_grad():
                        action, _ = model.select_action(obs_tensor, deterministic=True)
                    actions[agent_id] = action.item()

                # Step the environment with the chosen actions
                observations, _, terminations, _, _ = env.step(actions)

                # Check if the episode has ended for any agent
                terminated = any(terminations.values())

    except KeyboardInterrupt:
        print("Evaluation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model for 3v3 Soccer.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    args = parser.parse_args()
    evaluate(args)

import torch
import random
import copy

class SelfPlayManager:
    def __init__(self, initial_policy):
        """
        Manages policies for self-play training.

        Args:
            initial_policy: The initial policy network for the agents.
        """
        self.main_policy = initial_policy
        self.opponent_policy = copy.deepcopy(initial_policy)

    def get_main_policy(self):
        """
        Returns the main policy being trained.
        """
        return self.main_policy

    def get_opponent_policy(self):
        """
        Returns the opponent policy.
        """
        return self.opponent_policy

    def update_opponent_policy(self):
        """
        Updates the opponent policy with the weights of the main policy.
        """
        self.opponent_policy.load_state_dict(self.main_policy.state_dict())

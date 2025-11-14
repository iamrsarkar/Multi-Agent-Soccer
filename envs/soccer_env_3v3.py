import gymnasium
from gymnasium.spaces import Box, Discrete
import numpy as np
import pygame
from pettingzoo.utils.env import ParallelEnv

class SoccerEnv3v3(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "soccer_3v3"}

    def __init__(self, render_mode=None):
        self.possible_agents = [f"player_{i}" for i in range(6)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode

        self.screen_width = 800
        self.screen_height = 600
        self.ball = pygame.Rect(self.screen_width / 2 - 5, self.screen_height / 2 - 5, 10, 10)
        self.players = [pygame.Rect(0, 0, 20, 20) for _ in range(6)]
        self.player_velocities = [pygame.Vector2(0, 0) for _ in range(6)]
        self.ball_velocity = pygame.Vector2(0, 0)

        # Spaces
        self.observation_space = {
            agent: Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_space = {
            agent: Discrete(8) for agent in self.possible_agents
        }

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("3v3 Soccer")
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            player_pos = np.array([self.players[i].x, self.players[i].y], dtype=np.float32)
            player_vel = np.array([self.player_velocities[i].x, self.player_velocities[i].y], dtype=np.float32)
            ball_pos = np.array([self.ball.x, self.ball.y], dtype=np.float32)
            teammate_pos = np.array([p.center for j, p in enumerate(self.players) if j != i and (i < 3 and j < 3 or i >= 3 and j >= 3)], dtype=np.float32)
            opponent_pos = np.array([p.center for j, p in enumerate(self.players) if i < 3 and j >= 3 or i >= 3 and j < 3], dtype=np.float32)

            obs = np.concatenate([
                player_pos,
                player_vel,
                ball_pos,
                teammate_pos.flatten(),
                opponent_pos.flatten()
            ])
            observations[agent] = obs
        return observations

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)
        self.ball_velocity = pygame.Vector2(0, 0)

        # Reset player positions
        self.players[0].center = (100, 150)
        self.players[1].center = (100, 300)
        self.players[2].center = (100, 450)
        self.players[3].center = (700, 150)
        self.players[4].center = (700, 300)
        self.players[5].center = (700, 450)

        for i in range(6):
            self.player_velocities[i] = pygame.Vector2(0, 0)

        return self._get_obs(), {agent: {} for agent in self.agents}

    def step(self, actions):
        # Update player positions based on actions
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            if action == 0:  # Move Up
                self.player_velocities[i].y -= 1
            elif action == 1:  # Move Down
                self.player_velocities[i].y += 1
            elif action == 2:  # Move Left
                self.player_velocities[i].x -= 1
            elif action == 3:  # Move Right
                self.player_velocities[i].x += 1
            elif action == 4:  # Dash
                self.player_velocities[i] *= 1.5
            elif action == 5:  # Kick
                if self.players[i].colliderect(self.ball):
                    self.ball_velocity = (pygame.Vector2(self.ball.center) - pygame.Vector2(self.players[i].center)).normalize() * 10
            elif action == 6:  # Pass
                # For simplicity, pass is a weaker kick
                if self.players[i].colliderect(self.ball):
                    self.ball_velocity = (pygame.Vector2(self.ball.center) - pygame.Vector2(self.players[i].center)).normalize() * 5
            elif action == 7:  # Idle
                pass

            # Update player position
            self.players[i].move_ip(self.player_velocities[i])
            self.player_velocities[i] *= 0.9  # Friction

        # Update ball position
        self.ball.move_ip(self.ball_velocity)
        self.ball_velocity *= 0.95  # Friction

        # Boundary checks
        for player in self.players:
            player.clamp_ip(self.screen.get_rect())
        self.ball.clamp_ip(self.screen.get_rect())

        # Rewards and dones
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Goal check
        if self.ball.left <= 0:
            rewards = {agent: 1 if i >= 3 else -1 for i, agent in enumerate(self.agents)}
            terminations = {agent: True for agent in self.agents}
        elif self.ball.right >= self.screen_width:
            rewards = {agent: 1 if i < 3 else -1 for i, agent in enumerate(self.agents)}
            terminations = {agent: True for agent in self.agents}

        if all(terminations.values()):
            self.agents = []

        return self._get_obs(), rewards, terminations, truncations, {agent: {} for agent in self.agents}

    def render(self):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.screen.fill((0, 128, 0))

            # Draw goals
            pygame.draw.rect(self.screen, (255, 255, 255), (0, 200, 10, 200))
            pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width - 10, 200, 10, 200))

            for i, player in enumerate(self.players):
                color = (0, 0, 255) if i < 3 else (255, 0, 0)
                pygame.draw.rect(self.screen, color, player)

            pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

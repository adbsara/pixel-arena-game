"""Gymnasium environment: the enemy learns to catch the player."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pixel_arena.envs.scripted_player import WanderingPlayer
from pixel_arena.game import OBS_SIZE, Arena, enemy_observation
from pixel_arena.game import config as cfg

MAX_STEPS = 600          # the episode is cut off after this many frames

CATCH_REWARD = 5.0       # paid once, when the enemy catches the player
STEP_PENALTY = 0.01      # charged every frame, so wasting time costs
APPROACH_SCALE = 0.01    # reward per pixel of distance closed this frame


class ChaseEnv(gym.Env):
    """The agent controls the enemy. A scripted player wanders around."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(cfg.NUM_ACTIONS)
        self.arena: Arena | None = None
        self.player: WanderingPlayer | None = None
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random

        # The arena and the scripted player share Gymnasium's random
        # generator, so one seed reproduces the whole episode.
        self.arena = Arena(self.np_random)
        self.player = WanderingPlayer(self.np_random)
        self._steps = 0

        return enemy_observation(self.arena), {}

    def step(self, action):
        prev_dist = self.arena.distance()

        player_action = self.player.act(self.arena)
        caught = self.arena.step(player_action, int(action))
        self._steps += 1

        # Reward: pay for distance closed, charge for time, bonus on catch.
        reward = APPROACH_SCALE * (prev_dist - self.arena.distance())
        reward -= STEP_PENALTY
        if caught:
            reward += CATCH_REWARD

        terminated = caught
        truncated = not caught and self._steps >= MAX_STEPS

        # Stable-Baselines3 logs "is_success" as a success rate, which gives
        # us the real metric (catch rate) next to the reward curve.
        info = {"is_success": caught}

        return enemy_observation(self.arena), float(reward), terminated, truncated, info
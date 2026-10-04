"""Measure how well a policy plays ChaseEnv, independently of the reward."""

from collections.abc import Callable

import numpy as np

from pixel_arena.envs import ChaseEnv
from pixel_arena.game import config as cfg

# A policy is any function that maps an observation to an action.
Policy = Callable[[np.ndarray], int]


def evaluate(policy: Policy, episodes: int = 200, seed: int = 0) -> dict[str, float]:
    """Play full episodes and report catch rate, mean length and mean reward."""
    env = ChaseEnv()
    catches = 0
    lengths = []
    returns = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        catches += int(info["is_success"])
        lengths.append(steps)
        returns.append(total_reward)

    return {
        "catch_rate": catches / episodes,
        "mean_steps": float(np.mean(lengths)),
        "mean_reward": float(np.mean(returns)),
    }


def random_policy(rng: np.random.Generator) -> Policy:
    """An enemy that ignores what it sees and moves at random."""
    return lambda obs: int(rng.integers(cfg.NUM_ACTIONS))


if __name__ == "__main__":
    result = evaluate(random_policy(np.random.default_rng(0)))
    print("random enemy:", {k: round(v, 3) for k, v in result.items()})
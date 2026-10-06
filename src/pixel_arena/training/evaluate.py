"""Measure how well a policy plays ChaseEnv, independently of the reward."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from pixel_arena.envs import ChaseEnv
from pixel_arena.envs.scripted_player import PerimeterPlayer, WanderingPlayer
from pixel_arena.game import config as cfg
from pixel_arena.game.policy import NumpyPolicy

WEIGHTS_PATH = Path("models/chaser.npz")

# A policy is any function that maps an observation to an action.
Policy = Callable[[np.ndarray], int]


def evaluate(
    policy: Policy,
    episodes: int = 200,
    seed: int = 0,
    player_cls: type = WanderingPlayer,
) -> dict[str, float]:
    """Play full episodes and report catch rate, mean length and mean reward."""
    env = ChaseEnv(player_cls=player_cls)
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
    enemies = {"random": random_policy(np.random.default_rng(0))}
    if WEIGHTS_PATH.exists():
        enemies["trained"] = NumpyPolicy.load(WEIGHTS_PATH)

    players = {"wandering": WanderingPlayer, "perimeter": PerimeterPlayer}

    # Every enemy against every kind of player.
    for enemy_name, policy in enemies.items():
        for player_name, player_cls in players.items():
            result = evaluate(policy, player_cls=player_cls)
            rounded = {k: round(v, 3) for k, v in result.items()}
            print(f"{enemy_name:8} enemy vs {player_name:9} player: {rounded}")
"""Measure how the enemy's speed changes the game, to tune difficulty with data."""

from pathlib import Path

from pixel_arena.envs.scripted_player import PerimeterPlayer
from pixel_arena.game import config as cfg
from pixel_arena.game.policy import NumpyPolicy
from pixel_arena.training.evaluate import evaluate

WEIGHTS_PATH = Path("models/chaser.npz")

ENEMY_SPEEDS = [1.20, 1.25, 1.30, 1.35, 1.40]


def main() -> None:
    policy = NumpyPolicy.load(WEIGHTS_PATH)
    original_speed = cfg.ENEMY_SPEED
    print(f"player speed: {cfg.PLAYER_SPEED}")

    try:
        for speed in ENEMY_SPEEDS:
            # The arena reads this value every frame, so changing it here
            # changes the game for the episodes that follow.
            cfg.ENEMY_SPEED = speed
            result = evaluate(policy, player_cls=PerimeterPlayer)
            ratio = speed / cfg.PLAYER_SPEED
            print(
                f"enemy speed {speed:.2f} (ratio {ratio:.2f}): "
                f"catch_rate {result['catch_rate']:.3f}, "
                f"mean_steps {result['mean_steps']:.1f}"
            )
    finally:
        cfg.ENEMY_SPEED = original_speed


if __name__ == "__main__":
    main()
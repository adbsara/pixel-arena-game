"""What the enemy sees. Shared by training and by the playable game."""

import numpy as np

from pixel_arena.game import config as cfg
from pixel_arena.game.arena import Arena

OBS_SIZE = 6

_ARENA_SIZE = np.array([cfg.ARENA_W, cfg.ARENA_H], dtype=np.float32)


def enemy_observation(arena: Arena) -> np.ndarray:
    """Return the enemy's view of the arena as 6 numbers in [-1, 1]."""
    # Where the player is, relative to the enemy.
    rel = (arena.player_pos - arena.enemy_pos) / _ARENA_SIZE

    # The enemy's own position: -1 is the left/top wall, +1 the right/bottom.
    own = arena.enemy_pos / _ARENA_SIZE * 2.0 - 1.0

    # How the player moved last frame, so the enemy can anticipate.
    vel = arena.player_vel / cfg.PLAYER_SPEED

    # float32 rounding can push a value a hair past 1, so clip to the
    # range the observation space promises.
    obs = np.concatenate([rel, own, vel])
    return np.clip(obs, -1.0, 1.0).astype(np.float32)
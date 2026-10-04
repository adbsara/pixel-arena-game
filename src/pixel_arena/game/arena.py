"""The arena: one room, one player, one enemy."""

import numpy as np

from pixel_arena.game import config as cfg


class Arena:
    """Holds the game state and advances it one frame at a time."""

    def __init__(self, rng: np.random.Generator | None = None):
        # The random generator is passed in so the caller (later: Gymnasium)
        # controls the seed and runs are reproducible.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.enemy_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.reset()

    def _random_pos(self) -> np.ndarray:
        return self.rng.uniform(cfg.POS_LOW, cfg.POS_HIGH).astype(np.float32)

    def reset(self) -> None:
        """Place both entities at random spots, far enough apart."""
        while True:
            self.player_pos = self._random_pos()
            self.enemy_pos = self._random_pos()
            if self.distance() >= cfg.MIN_SPAWN_DIST:
                break
        self.player_vel = np.zeros(2, dtype=np.float32)

    def distance(self) -> float:
        """Straight-line distance between the two centers."""
        return float(np.linalg.norm(self.player_pos - self.enemy_pos))

    def is_caught(self) -> bool:
        """Two equal squares overlap when both axis gaps are below one side."""
        dx, dy = np.abs(self.player_pos - self.enemy_pos)
        return bool(dx < cfg.ENTITY_SIZE and dy < cfg.ENTITY_SIZE)

    def step(self, player_action: int, enemy_action: int) -> bool:
        """Advance one frame. Returns True if the enemy caught the player."""
        prev_player = self.player_pos.copy()

        self.player_pos = np.clip(
            self.player_pos + cfg.DIRECTIONS[player_action] * cfg.PLAYER_SPEED,
            cfg.POS_LOW, cfg.POS_HIGH,
        )
        self.enemy_pos = np.clip(
            self.enemy_pos + cfg.DIRECTIONS[enemy_action] * cfg.ENEMY_SPEED,
            cfg.POS_LOW, cfg.POS_HIGH,
        )

        # Real displacement after clipping: zero when pushing into a wall.
        self.player_vel = self.player_pos - prev_player

        return self.is_caught()
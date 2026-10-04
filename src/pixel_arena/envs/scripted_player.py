"""Scripted stand-ins for the human player, used only during training."""

import numpy as np

from pixel_arena.game import config as cfg
from pixel_arena.game.arena import Arena

MIN_HOLD = 10   # shortest time (in frames) to keep one direction
MAX_HOLD = 40   # longest time (in frames) to keep one direction


class WanderingPlayer:
    """Picks a random direction and keeps it for a random number of frames."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._action = cfg.STAY
        self._frames_left = 0

    def reset(self) -> None:
        """Forget the current direction. Call at the start of each episode."""
        self._frames_left = 0

    def act(self, arena: Arena) -> int:
        """Return the player's action for this frame."""
        # `arena` is unused for now. A smarter player (one that runs away
        # from the enemy) will need it, so the interface already takes it.
        if self._frames_left <= 0:
            self._action = int(self.rng.integers(cfg.NUM_ACTIONS))
            self._frames_left = int(self.rng.integers(MIN_HOLD, MAX_HOLD + 1))
        self._frames_left -= 1
        return self._action
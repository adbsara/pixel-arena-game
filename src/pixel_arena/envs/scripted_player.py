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
        if self._frames_left <= 0:
            self._action = int(self.rng.integers(cfg.NUM_ACTIONS))
            self._frames_left = int(self.rng.integers(MIN_HOLD, MAX_HOLD + 1))
        self._frames_left -= 1
        return self._action


# --- The track along the walls ----------------------------------------------
# The four walls form a rectangular track. A place on the track is one
# number: the distance travelled clockwise from the top-left corner.

_X0, _Y0 = float(cfg.POS_LOW[0]), float(cfg.POS_LOW[1])
_X1, _Y1 = float(cfg.POS_HIGH[0]), float(cfg.POS_HIGH[1])
_W = _X1 - _X0
_H = _Y1 - _Y0
_TRACK = 2 * (_W + _H)

_ON_WALL = 0.01        # closer than this counts as touching the wall
SWITCH_MARGIN = 40.0   # turn around only if the enemy is clearly closer ahead


def _wall_distances(pos: np.ndarray) -> list[float]:
    """Distance to the top, right, bottom and left wall, in that order."""
    x, y = float(pos[0]), float(pos[1])
    return [y - _Y0, _X1 - x, _Y1 - y, x - _X0]


def _track_position(pos: np.ndarray) -> float:
    """Project a point onto its nearest wall and return its place on the track."""
    x, y = float(pos[0]), float(pos[1])
    wall = int(np.argmin(_wall_distances(pos)))
    if wall == 0:                        # top wall, clockwise is rightwards
        return x - _X0
    if wall == 1:                        # right wall, clockwise is downwards
        return _W + (y - _Y0)
    if wall == 2:                        # bottom wall, clockwise is leftwards
        return _W + _H + (_X1 - x)
    return 2 * _W + _H + (_Y1 - y)       # left wall, clockwise is upwards


class PerimeterPlayer:
    """Runs along the walls, away from the enemy.

    This copies an exploit found by a human player: hug the walls and
    circle the arena, turning around whenever the enemy gets ahead.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._clockwise = True
        self.reset()

    def reset(self) -> None:
        """Pick a random direction of travel for the new episode."""
        self._clockwise = bool(self.rng.integers(2))

    def act(self, arena: Arena) -> int:
        """Return the player's action for this frame."""
        distances = _wall_distances(arena.player_pos)
        nearest = int(np.argmin(distances))

        # Not on a wall yet: head straight for the nearest one.
        if distances[nearest] > _ON_WALL:
            return (cfg.UP, cfg.RIGHT, cfg.DOWN, cfg.LEFT)[nearest]

        # On the track: is the enemy closer ahead of us or behind us?
        me = _track_position(arena.player_pos)
        enemy = _track_position(arena.enemy_pos)
        clockwise_gap = (enemy - me) % _TRACK
        ahead = clockwise_gap if self._clockwise else _TRACK - clockwise_gap
        behind = _TRACK - ahead
        if ahead < behind - SWITCH_MARGIN:
            self._clockwise = not self._clockwise

        return self._along_track(me)

    def _along_track(self, place: float) -> int:
        """The action that follows the walls in the current direction."""
        if self._clockwise:
            if place < _W:
                return cfg.RIGHT             # top wall
            if place < _W + _H:
                return cfg.DOWN              # right wall
            if place < 2 * _W + _H:
                return cfg.LEFT              # bottom wall
            return cfg.UP                    # left wall

        if place == 0 or place > 2 * _W + _H:
            return cfg.DOWN                  # left wall
        if place <= _W:
            return cfg.LEFT                  # top wall
        if place <= _W + _H:
            return cfg.UP                    # right wall
        return cfg.RIGHT                     # bottom wall
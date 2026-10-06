"""Game constants. Every tunable number of the game rules lives here."""

import numpy as np

# --- Arena -----------------------------------------------------------------

ARENA_W = 160          # arena width in logical pixels
ARENA_H = 120          # arena height in logical pixels
ENTITY_SIZE = 8        # side of the player / enemy square
MIN_SPAWN_DIST = 90    # minimum distance between the two at reset

# --- Movement --------------------------------------------------------------

PLAYER_SPEED = 1.6     # pixels per frame
ENEMY_SPEED = 1.3    # pixels per frame

# --- Actions ---------------------------------------------------------------

STAY, UP, DOWN, LEFT, RIGHT = range(5)

# Action index -> (dx, dy). Screen y grows downward, so UP is -1.
DIRECTIONS = np.array(
    [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]],
    dtype=np.float32,
)
NUM_ACTIONS = len(DIRECTIONS)

# --- Position bounds -------------------------------------------------------

# Positions are the CENTER of each square, so the center must stay
# half a square away from every wall.
_HALF = ENTITY_SIZE / 2
POS_LOW = np.array([_HALF, _HALF], dtype=np.float32)
POS_HIGH = np.array([ARENA_W - _HALF, ARENA_H - _HALF], dtype=np.float32)
"""Game constants. Every tunable number of the game rules lives here."""

import numpy as np

# --- Arena -------------------------------------------------------------------

TILE = 16                # one floor tile; the art pack uses 16x16 tiles
ARENA_W = 20 * TILE      # arena width in logical pixels (320)
ARENA_H = 13 * TILE      # arena height in logical pixels (208)
ENTITY_SIZE = 12         # side of the player / enemy square
MIN_SPAWN_DIST = 160     # minimum distance between the two at reset

# --- Movement ----------------------------------------------------------------

PLAYER_SPEED = 1.5       # pixels per frame
ENEMY_SPEED = 1.05       # pixels per frame

# --- Actions -----------------------------------------------------------------

STAY, UP, DOWN, LEFT, RIGHT = range(5)

# Action index -> (dx, dy). Screen y grows downward, so UP is -1.
DIRECTIONS = np.array(
    [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]],
    dtype=np.float32,
)
NUM_ACTIONS = len(DIRECTIONS)

# --- Position bounds ---------------------------------------------------------

# Positions are the CENTER of each square, so the center must stay
# half a square away from every wall.
_HALF = ENTITY_SIZE / 2
POS_LOW = np.array([_HALF, _HALF], dtype=np.float32)
POS_HIGH = np.array([ARENA_W - _HALF, ARENA_H - _HALF], dtype=np.float32)

# --- Coins -------------------------------------------------------------------

COIN_SIZE = 8              # side of the coin square
COIN_WALL_MARGIN = 40      # coins never appear closer than this to a wall
COIN_MIN_DIST = 70         # a new coin appears at least this far from the player
COIN_ENEMY_MIN_DIST = 90   # ... and at least this far from the enemy
COIN_CANDIDATES = 3 
COIN_LOW = np.array([COIN_WALL_MARGIN, COIN_WALL_MARGIN], dtype=np.float32)
COIN_HIGH = np.array(
    [ARENA_W - COIN_WALL_MARGIN, ARENA_H - COIN_WALL_MARGIN], dtype=np.float32
)

# --- Lives -------------------------------------------------------------------

START_LIVES = 3            # hits the player can take before the round ends
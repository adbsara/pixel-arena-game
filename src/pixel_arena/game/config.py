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


# --- Combat ------------------------------------------------------------------

ATTACK_RANGE = 34          # how far to either side the swing still connects
ATTACK_COOLDOWN = 40       # frames the player must wait between swings
STUN_FRAMES = 90           # frames a struck enemy stays frozen and harmless
KNOCKBACK = 14 
MAX_SWORDS = 5                 # pixels a struck enemy is pushed back


# --- Obstacles -----------------------------------------------------------

# Each wall is a rectangle: (x, y, width, height), in arena coordinates.
# This layout is one horizontal wall across the middle, with a gap at
# each end wide enough to walk through.
_WALL_Y = ARENA_H // 2 - TILE // 2
_GAP = TILE * 3
WALLS = [
    (0, _WALL_Y, ARENA_W // 2 - _GAP, TILE),
    (ARENA_W // 2 + _GAP, _WALL_Y, ARENA_W - (ARENA_W // 2 + _GAP), TILE),
]
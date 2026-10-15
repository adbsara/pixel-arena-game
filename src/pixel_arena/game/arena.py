"""The arena: one room, one player, one enemy, one coin."""

import numpy as np

from pixel_arena.game import config as cfg

_COIN_TRIES = 200   # give up looking for a valid coin spot after this many


class Arena:
    """Holds the game state and advances it one frame at a time."""

    def __init__(self, rng: np.random.Generator | None = None):
        # The random generator is passed in so the caller (later: Gymnasium)
        # controls the seed and runs are reproducible.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.enemy_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.coin_pos = np.zeros(2, dtype=np.float32)
        self.score = 0                # coins collected this round
        self.lives = cfg.START_LIVES  # hits left before the round ends
        self.just_collected = False   # True only on the frame a coin is taken

        # Combat state.
        self.facing = cfg.RIGHT       # the last direction the player moved in
        self.attack_cooldown = 0      # frames until the player may swing again
        self.enemy_stun = 0           # frames the enemy stays frozen
        self.swords = cfg.MAX_SWORDS  # swings left this round
        self.just_attacked = False    # True only on the frame of a swing
        self.just_struck = False      # True only on the frame a swing connects

        self.reset()

    def _random_pos(self) -> np.ndarray:
        return self.rng.uniform(cfg.POS_LOW, cfg.POS_HIGH).astype(np.float32)

    def _valid_coin_spot(self) -> np.ndarray:
        """A spot away from the walls, the player and the enemy."""
        pos = self.coin_pos
        for _ in range(_COIN_TRIES):
            pos = self.rng.uniform(cfg.COIN_LOW, cfg.COIN_HIGH).astype(np.float32)
            from_player = np.linalg.norm(pos - self.player_pos)
            from_enemy = np.linalg.norm(pos - self.enemy_pos)
            if from_player >= cfg.COIN_MIN_DIST and from_enemy >= cfg.COIN_ENEMY_MIN_DIST:
                break
        return pos

    def _random_coin_pos(self) -> np.ndarray:
        """Try a few valid spots and keep the one that favours the player most.

        A spot favours the player when it is nearer to the player than to
        the enemy. Without this, new coins tend to land behind the enemy,
        because the enemy is usually right behind the player at pick-up.
        """
        best = self.coin_pos
        best_edge = -np.inf
        for _ in range(cfg.COIN_CANDIDATES):
            pos = self._valid_coin_spot()
            edge = np.linalg.norm(pos - self.enemy_pos) - np.linalg.norm(
                pos - self.player_pos
            )
            if edge > best_edge:
                best, best_edge = pos, edge
        return best

    def reset(self) -> None:
        """Start a new round: fresh positions, a fresh coin, full lives."""
        while True:
            self.player_pos = self._random_pos()
            self.enemy_pos = self._random_pos()
            if self.distance() >= cfg.MIN_SPAWN_DIST:
                break
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.coin_pos = self._random_coin_pos()
        self.score = 0
        self.lives = cfg.START_LIVES
        self.just_collected = False
        self.facing = cfg.RIGHT
        self.attack_cooldown = 0
        self.enemy_stun = 0
        self.swords = cfg.MAX_SWORDS
        self.just_attacked = False
        self.just_struck = False

    def distance(self) -> float:
        """Straight-line distance between the player and the enemy."""
        return float(np.linalg.norm(self.player_pos - self.enemy_pos))

    def is_caught(self) -> bool:
        """Two equal squares overlap when both axis gaps are below one side."""
        dx, dy = np.abs(self.player_pos - self.enemy_pos)
        return bool(dx < cfg.ENTITY_SIZE and dy < cfg.ENTITY_SIZE)

    def touches_coin(self) -> bool:
        """The player's square overlaps the coin's square."""
        dx, dy = np.abs(self.player_pos - self.coin_pos)
        reach = (cfg.ENTITY_SIZE + cfg.COIN_SIZE) / 2
        return bool(dx < reach and dy < reach)

    def sword_reaches_enemy(self) -> bool:
        """Any enemy within range is hit, regardless of which way the player faces.

        Precise directional aiming is hard with keyboard controls at this
        speed, so the hitbox is a circle around the player rather than a
        narrow cone. The sword is still drawn pointing the way the player
        last moved, for feedback, but the hit itself is forgiving.
        """
        return self.distance() <= cfg.ATTACK_RANGE

    def send_enemy_away(self) -> None:
        """Move the enemy to the corner farthest from the player.

        Called by the game after a hit, so the player gets room to recover.
        It is not part of step(), so training episodes are not affected.
        """
        corners = [
            (x, y)
            for x in (cfg.POS_LOW[0], cfg.POS_HIGH[0])
            for y in (cfg.POS_LOW[1], cfg.POS_HIGH[1])
        ]
        px, py = self.player_pos
        far = max(corners, key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        self.enemy_pos = np.array(far, dtype=np.float32)

    def step(self, player_action: int, enemy_action: int, attack: bool = False) -> bool:
        """Advance one frame. Returns True if the enemy hit the player."""
        prev_player = self.player_pos.copy()

        if player_action != cfg.STAY:
            self.facing = player_action

        self.player_pos = np.clip(
            self.player_pos + cfg.DIRECTIONS[player_action] * cfg.PLAYER_SPEED,
            cfg.POS_LOW, cfg.POS_HIGH,
        )

        # A stunned enemy stands still until the stun wears off.
        if self.enemy_stun > 0:
            self.enemy_stun -= 1
        else:
            self.enemy_pos = np.clip(
                self.enemy_pos + cfg.DIRECTIONS[enemy_action] * cfg.ENEMY_SPEED,
                cfg.POS_LOW, cfg.POS_HIGH,
            )

        # Real displacement after clipping: zero when pushing into a wall.
        self.player_vel = self.player_pos - prev_player

        # The swing. It only happens if the cooldown has run out and at
        # least one sword charge is left.
        self.just_attacked = False
        self.just_struck = False
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        elif attack and self.swords > 0:
            self.just_attacked = True
            self.attack_cooldown = cfg.ATTACK_COOLDOWN
            self.swords -= 1
            if self.sword_reaches_enemy():
                self.just_struck = True
                self.enemy_stun = cfg.STUN_FRAMES
                # Knock the enemy straight back, away from the player.
                gap = self.enemy_pos - self.player_pos
                norm = float(np.linalg.norm(gap))
                direction = gap / norm if norm > 1e-6 else cfg.DIRECTIONS[self.facing]
                self.enemy_pos = np.clip(
                    self.enemy_pos + direction * cfg.KNOCKBACK,
                    cfg.POS_LOW, cfg.POS_HIGH,
                )

        self.just_collected = self.touches_coin()
        if self.just_collected:
            self.score += 1
            self.coin_pos = self._random_coin_pos()

        # A stunned enemy cannot hurt the player.
        hit = self.is_caught() and self.enemy_stun == 0
        if hit:
            self.lives = max(0, self.lives - 1)
        return hit
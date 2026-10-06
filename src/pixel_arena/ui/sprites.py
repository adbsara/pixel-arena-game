"""Loads the art pack and names the tiles the game uses."""

from pathlib import Path

import numpy as np
import pygame

from pixel_arena.game import config as cfg

SHEET_PATH = Path("assets/tiny_dungeon/tilemap_packed.png")

# Tile numbers, as shown by the tile viewer. Change these to restyle the game.
PLAYER_TILE = 88
ENEMY_TILE = 110
COIN_TILE = 114
WALL_TILE = 40
FLOOR_TILE = 48
FLOOR_DETAIL_TILES = (49, 42)   # sprinkled over the floor for variety
FLOOR_DETAIL_CHANCE = 0.12      # share of floor tiles that get a detail

# A ring of wall tiles is drawn around the arena, outside the play area,
# so the rules of the game do not change.
BORDER = cfg.TILE
CANVAS_SIZE = (cfg.ARENA_W + 2 * BORDER, cfg.ARENA_H + 2 * BORDER)


class Sprites:
    """Every image the game draws. Create it after the window exists."""

    def __init__(self):
        self._sheet = pygame.image.load(SHEET_PATH).convert_alpha()
        self._cols = self._sheet.get_width() // cfg.TILE

        self.player = self.tile(PLAYER_TILE)
        self.enemy = self.tile(ENEMY_TILE)
        self.coin = self.tile(COIN_TILE)
        self.background = self._build_background()

    def tile(self, index: int) -> pygame.Surface:
        """Cut tile number `index` out of the sheet."""
        col, row = index % self._cols, index // self._cols
        area = (col * cfg.TILE, row * cfg.TILE, cfg.TILE, cfg.TILE)
        return self._sheet.subsurface(area).copy()

    def _build_background(self) -> pygame.Surface:
        """Floor and walls, drawn once and reused every frame."""
        surface = pygame.Surface(CANVAS_SIZE)
        wall = self.tile(WALL_TILE)
        floor = self.tile(FLOOR_TILE)
        details = [self.tile(index) for index in FLOOR_DETAIL_TILES]

        # A fixed seed gives the same floor on every run.
        rng = np.random.default_rng(7)
        cols = CANVAS_SIZE[0] // cfg.TILE
        rows = CANVAS_SIZE[1] // cfg.TILE

        for row in range(rows):
            for col in range(cols):
                on_border = row in (0, rows - 1) or col in (0, cols - 1)
                if on_border:
                    image = wall
                elif rng.random() < FLOOR_DETAIL_CHANCE:
                    image = details[int(rng.integers(len(details)))]
                else:
                    image = floor
                surface.blit(image, (col * cfg.TILE, row * cfg.TILE))
        return surface

    def draw(self, canvas: pygame.Surface, image: pygame.Surface, pos) -> None:
        """Draw a tile centred on a game position."""
        half = cfg.TILE // 2
        canvas.blit(image, (int(pos[0]) + BORDER - half, int(pos[1]) + BORDER - half))
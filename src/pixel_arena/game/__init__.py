"""Pure game logic. Nothing in this package may import pygame."""

from pixel_arena.game.arena import Arena
from pixel_arena.game.observation import OBS_SIZE, enemy_observation

__all__ = ["Arena", "OBS_SIZE", "enemy_observation"]
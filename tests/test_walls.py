"""Tests for interior walls: spawning, movement and sliding."""

import numpy as np

from pixel_arena.game import Arena
from pixel_arena.game import config as cfg


def make_arena(seed: int = 0) -> Arena:
    return Arena(np.random.default_rng(seed))


def test_spawn_positions_never_land_inside_a_wall():
    arena = make_arena()
    half = cfg.ENTITY_SIZE / 2
    for _ in range(100):
        arena.reset()
        for pos in (arena.player_pos, arena.enemy_pos):
            for wx, wy, ww, wh in cfg.WALLS:
                inside = (
                    wx - half < pos[0] < wx + ww + half
                    and wy - half < pos[1] < wy + wh + half
                )
                assert not inside


def test_coin_never_lands_inside_a_wall():
    arena = make_arena()
    half = cfg.COIN_SIZE / 2
    for _ in range(100):
        arena.reset()
        for wx, wy, ww, wh in cfg.WALLS:
            inside = (
                wx - half < arena.coin_pos[0] < wx + ww + half
                and wy - half < arena.coin_pos[1] < wy + wh + half
            )
            assert not inside


def test_player_cannot_walk_straight_through_a_wall():
    arena = make_arena()
    wx, wy, ww, wh = cfg.WALLS[0]
    # Just above the wall, moving straight down into it.
    arena.player_pos = np.array([wx + 5, wy - cfg.ENTITY_SIZE], dtype=np.float32)

    for _ in range(60):
        arena.step(cfg.DOWN, cfg.STAY)

    # Blocked: the player stayed above the wall's top edge.
    assert arena.player_pos[1] < wy


def test_player_slides_along_a_wall_on_a_diagonal_approach():
    arena = make_arena()
    wx, wy, ww, wh = cfg.WALLS[0]
    start_x = wx + 5
    arena.player_pos = np.array([start_x, wy - cfg.ENTITY_SIZE], dtype=np.float32)

    # Moving down-right: blocked vertically by the wall, but free to slide
    # sideways, so x should still change even though y does not cross wy.
    for _ in range(10):
        arena.step(cfg.DOWN, cfg.STAY)
    for _ in range(10):
        arena.step(cfg.RIGHT, cfg.STAY)

    assert arena.player_pos[0] > start_x


def test_player_can_walk_through_the_gap():
    arena = make_arena()
    wx, wy, ww, wh = cfg.WALLS[0]
    gap_x = ww + (cfg.WALLS[1][0] - ww) / 2   # middle of the gap
    arena.player_pos = np.array([gap_x, wy - cfg.ENTITY_SIZE], dtype=np.float32)

    for _ in range(60):
        arena.step(cfg.DOWN, cfg.STAY)

    # Through: the player made it below the wall's bottom edge.
    assert arena.player_pos[1] > wy + wh
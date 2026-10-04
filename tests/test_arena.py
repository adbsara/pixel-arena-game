"""Tests for the pure game logic."""

import numpy as np

from pixel_arena.game import Arena
from pixel_arena.game import config as cfg


def make_arena(seed: int = 0) -> Arena:
    return Arena(np.random.default_rng(seed))


def test_reset_keeps_minimum_distance():
    arena = make_arena()
    for _ in range(200):
        arena.reset()
        assert arena.distance() >= cfg.MIN_SPAWN_DIST


def test_positions_stay_inside_arena():
    rng = np.random.default_rng(1)
    arena = Arena(rng)
    for _ in range(5000):
        arena.step(
            int(rng.integers(cfg.NUM_ACTIONS)),
            int(rng.integers(cfg.NUM_ACTIONS)),
        )
        for pos in (arena.player_pos, arena.enemy_pos):
            assert np.all(pos >= cfg.POS_LOW)
            assert np.all(pos <= cfg.POS_HIGH)


def test_overlapping_squares_are_caught():
    arena = make_arena()
    arena.player_pos = np.array([50, 50], dtype=np.float32)
    arena.enemy_pos = np.array([50 + cfg.ENTITY_SIZE - 1, 50], dtype=np.float32)
    assert arena.step(cfg.STAY, cfg.STAY) is True


def test_separated_squares_are_not_caught():
    arena = make_arena()
    arena.player_pos = np.array([50, 50], dtype=np.float32)
    arena.enemy_pos = np.array([50 + cfg.ENTITY_SIZE + 5, 50], dtype=np.float32)
    assert arena.step(cfg.STAY, cfg.STAY) is False


def test_velocity_is_zero_when_pushing_into_wall():
    arena = make_arena()
    arena.player_pos = cfg.POS_LOW.copy()
    arena.enemy_pos = cfg.POS_HIGH.copy()
    arena.step(cfg.LEFT, cfg.STAY)
    assert np.allclose(arena.player_vel, 0.0)


def test_player_moves_at_player_speed():
    arena = make_arena()
    arena.player_pos = np.array([80, 60], dtype=np.float32)
    arena.enemy_pos = cfg.POS_LOW.copy()
    arena.step(cfg.RIGHT, cfg.STAY)
    assert np.allclose(arena.player_vel, [cfg.PLAYER_SPEED, 0.0])


def test_same_seed_gives_same_start():
    a, b = make_arena(42), make_arena(42)
    assert np.array_equal(a.player_pos, b.player_pos)
    assert np.array_equal(a.enemy_pos, b.enemy_pos)
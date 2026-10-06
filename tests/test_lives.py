"""Tests for lives and for coin placement relative to the enemy."""

import numpy as np

from pixel_arena.game import Arena
from pixel_arena.game import config as cfg


def make_arena(seed: int = 0) -> Arena:
    return Arena(np.random.default_rng(seed))


def test_round_starts_with_full_lives():
    arena = make_arena()
    assert arena.lives == cfg.START_LIVES


def test_a_hit_costs_one_life():
    arena = make_arena()
    arena.player_pos = np.array([50, 50], dtype=np.float32)
    arena.enemy_pos = np.array([52, 50], dtype=np.float32)

    hit = arena.step(cfg.STAY, cfg.STAY)

    assert hit is True
    assert arena.lives == cfg.START_LIVES - 1


def test_enemy_is_sent_to_the_farthest_corner():
    arena = make_arena()
    arena.player_pos = cfg.POS_LOW.copy()
    arena.send_enemy_away()
    assert np.array_equal(arena.enemy_pos, cfg.POS_HIGH)


def test_coin_keeps_its_distance_from_the_enemy():
    arena = make_arena()
    for _ in range(200):
        arena.reset()
        gap = np.linalg.norm(arena.coin_pos - arena.enemy_pos)
        assert gap >= cfg.COIN_ENEMY_MIN_DIST


def test_reset_restores_lives():
    arena = make_arena()
    arena.lives = 1
    arena.reset()
    assert arena.lives == cfg.START_LIVES
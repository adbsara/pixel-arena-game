"""Tests for the coin rules."""

import numpy as np

from pixel_arena.game import Arena
from pixel_arena.game import config as cfg


def make_arena(seed: int = 0) -> Arena:
    return Arena(np.random.default_rng(seed))


def test_coin_appears_away_from_walls_and_player():
    arena = make_arena()
    for _ in range(200):
        arena.reset()
        assert np.all(arena.coin_pos >= cfg.COIN_LOW)
        assert np.all(arena.coin_pos <= cfg.COIN_HIGH)
        gap = np.linalg.norm(arena.coin_pos - arena.player_pos)
        assert gap >= cfg.COIN_MIN_DIST


def test_touching_the_coin_scores_and_moves_it():
    arena = make_arena()
    arena.player_pos = np.array([80, 60], dtype=np.float32)
    arena.enemy_pos = cfg.POS_LOW.copy()
    arena.coin_pos = arena.player_pos.copy()

    arena.step(cfg.STAY, cfg.STAY)

    assert arena.score == 1
    assert arena.just_collected is True
    gap = np.linalg.norm(arena.coin_pos - arena.player_pos)
    assert gap >= cfg.COIN_MIN_DIST


def test_far_coin_is_not_collected():
    arena = make_arena()
    arena.player_pos = np.array([130, 90], dtype=np.float32)
    arena.enemy_pos = cfg.POS_LOW.copy()
    arena.coin_pos = np.array([30, 30], dtype=np.float32)

    arena.step(cfg.STAY, cfg.STAY)

    assert arena.score == 0
    assert arena.just_collected is False


def test_reset_clears_the_score():
    arena = make_arena()
    arena.score = 5
    arena.reset()
    assert arena.score == 0
    

def test_coin_usually_favours_the_player():
    arena = make_arena()
    rounds = 300
    closer_to_player = 0
    for _ in range(rounds):
        arena.reset()
        to_player = np.linalg.norm(arena.coin_pos - arena.player_pos)
        to_enemy = np.linalg.norm(arena.coin_pos - arena.enemy_pos)
        closer_to_player += int(to_player < to_enemy)
    assert closer_to_player / rounds > 0.6
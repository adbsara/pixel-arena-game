"""Tests for the sword: reach, stun, knockback and cooldown."""

import numpy as np

from pixel_arena.game import Arena
from pixel_arena.game import config as cfg


def make_arena(seed: int = 0) -> Arena:
    return Arena(np.random.default_rng(seed))


def place(arena: Arena, player: tuple, enemy: tuple) -> None:
    arena.player_pos = np.array(player, dtype=np.float32)
    arena.enemy_pos = np.array(enemy, dtype=np.float32)


def test_a_swing_at_an_enemy_in_front_stuns_and_pushes_it():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(120, 100))

    arena.step(cfg.RIGHT, cfg.STAY, attack=True)

    assert arena.just_struck is True
    assert arena.enemy_stun == cfg.STUN_FRAMES
    assert np.isclose(arena.enemy_pos[0], 120 + cfg.KNOCKBACK)


def test_a_swing_misses_an_enemy_behind_the_player():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(80, 100))

    arena.step(cfg.RIGHT, cfg.STAY, attack=True)

    assert arena.just_attacked is True
    assert arena.just_struck is False
    assert arena.enemy_stun == 0


def test_a_stunned_enemy_does_not_move():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(160, 100))
    arena.enemy_stun = 10

    arena.step(cfg.STAY, cfg.LEFT)

    assert np.isclose(arena.enemy_pos[0], 160)
    assert arena.enemy_stun == 9


def test_a_stunned_enemy_cannot_hurt_the_player():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(102, 100))
    arena.enemy_stun = 10

    hit = arena.step(cfg.STAY, cfg.STAY)

    assert hit is False
    assert arena.lives == cfg.START_LIVES


def test_the_cooldown_blocks_a_second_swing():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(250, 180))

    arena.step(cfg.STAY, cfg.STAY, attack=True)
    assert arena.just_attacked is True

    arena.step(cfg.STAY, cfg.STAY, attack=True)
    assert arena.just_attacked is False
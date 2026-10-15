"""Tests for the sword: range, stun, knockback and cooldown."""

import numpy as np

from pixel_arena.game import Arena
from pixel_arena.game import config as cfg


def make_arena(seed: int = 0) -> Arena:
    return Arena(np.random.default_rng(seed))


def place(arena: Arena, player: tuple, enemy: tuple) -> None:
    arena.player_pos = np.array(player, dtype=np.float32)
    arena.enemy_pos = np.array(enemy, dtype=np.float32)


def test_a_swing_at_a_nearby_enemy_stuns_and_pushes_it_away():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(120, 100))

    arena.step(cfg.RIGHT, cfg.STAY, attack=True)

    assert arena.just_struck is True
    assert arena.enemy_stun == cfg.STUN_FRAMES
    assert np.isclose(arena.enemy_pos[0], 120 + cfg.KNOCKBACK)


def test_a_swing_hits_a_nearby_enemy_in_any_direction():
    # Facing right, but the enemy is close behind, to the left.
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(80, 100))

    arena.step(cfg.RIGHT, cfg.STAY, attack=True)

    assert arena.just_struck is True
    # Knockback pushes the enemy away from the player (leftwards), or at
    # least never back toward it.
    assert arena.enemy_pos[0] <= 80

def test_a_swing_misses_a_distant_enemy():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(300, 100))

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



def test_round_starts_with_full_swords():
    arena = make_arena()
    assert arena.swords == cfg.MAX_SWORDS


def test_running_out_of_swords_blocks_further_swings():
    arena = make_arena()
    place(arena, player=(100, 100), enemy=(300, 100))

    for _ in range(cfg.MAX_SWORDS):
        arena.step(cfg.STAY, cfg.STAY, attack=True)
        # Wait out the cooldown so the next press is a fresh swing attempt.
        for _ in range(cfg.ATTACK_COOLDOWN):
            arena.step(cfg.STAY, cfg.STAY)

    assert arena.swords == 0

    arena.step(cfg.STAY, cfg.STAY, attack=True)
    assert arena.just_attacked is False
"""Play against the trained enemy. Arrow keys or WASD to move, Esc to quit."""

import asyncio
from pathlib import Path

import numpy as np
import pygame

from pixel_arena.game import Arena, enemy_observation
from pixel_arena.game import config as cfg
from pixel_arena.game.policy import NumpyPolicy

WEIGHTS_PATH = Path("models/chaser.npz")

SCALE = 5      # each logical pixel becomes a 5x5 block on screen
FPS = 60
WINDOW_SIZE = (cfg.ARENA_W * SCALE, cfg.ARENA_H * SCALE)

BACKGROUND = (24, 26, 38)
PLAYER_COLOR = (92, 200, 255)
ENEMY_COLOR = (255, 96, 96)
TEXT_COLOR = (230, 230, 240)

# The first pressed key in this list wins. The player has the same five
# actions as the enemy, so there is no diagonal movement.
KEY_ACTIONS = [
    ((pygame.K_UP, pygame.K_w), cfg.UP),
    ((pygame.K_DOWN, pygame.K_s), cfg.DOWN),
    ((pygame.K_LEFT, pygame.K_a), cfg.LEFT),
    ((pygame.K_RIGHT, pygame.K_d), cfg.RIGHT),
]


def read_player_action() -> int:
    pressed = pygame.key.get_pressed()
    for keys, action in KEY_ACTIONS:
        if any(pressed[key] for key in keys):
            return action
    return cfg.STAY


def draw_entity(canvas: pygame.Surface, pos: np.ndarray, color: tuple) -> None:
    half = cfg.ENTITY_SIZE // 2
    rect = pygame.Rect(
        int(pos[0]) - half, int(pos[1]) - half, cfg.ENTITY_SIZE, cfg.ENTITY_SIZE
    )
    pygame.draw.rect(canvas, color, rect)


async def main() -> None:
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pixel Arena")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    # Everything is drawn on a tiny canvas, then blown up without smoothing.
    # That is what gives the chunky pixel look.
    canvas = pygame.Surface((cfg.ARENA_W, cfg.ARENA_H))

    arena = Arena(np.random.default_rng())
    enemy = NumpyPolicy.load(WEIGHTS_PATH)

    catches = 0
    frames_alive = 0
    best_frames = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # One frame of the game: you pick your action, the network picks
        # the enemy's from the same observation it was trained on.
        enemy_action = enemy(enemy_observation(arena))
        caught = arena.step(read_player_action(), enemy_action)

        frames_alive += 1
        if caught:
            catches += 1
            best_frames = max(best_frames, frames_alive)
            frames_alive = 0
            arena.reset()

        canvas.fill(BACKGROUND)
        draw_entity(canvas, arena.player_pos, PLAYER_COLOR)
        draw_entity(canvas, arena.enemy_pos, ENEMY_COLOR)
        window.blit(pygame.transform.scale(canvas, WINDOW_SIZE), (0, 0))

        hud = (
            f"caught: {catches}   "
            f"survived: {frames_alive / FPS:.1f}s   "
            f"best: {best_frames / FPS:.1f}s"
        )
        window.blit(font.render(hud, True, TEXT_COLOR), (10, 8))

        pygame.display.flip()
        clock.tick(FPS)

        # Hands control back for a moment. Needed later to run in a browser.
        await asyncio.sleep(0)

    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
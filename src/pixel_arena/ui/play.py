"""Play against the trained enemy. Arrow keys or WASD to move, Esc to quit."""

import asyncio
from pathlib import Path

import numpy as np
import pygame

from pixel_arena.game import Arena, enemy_observation
from pixel_arena.game import config as cfg
from pixel_arena.game.policy import NumpyPolicy

WEIGHTS_PATH = Path("models/chaser.npz")

SCALE = 5            # each logical pixel becomes a 5x5 block on screen
FPS = 60
PAUSE_FRAMES = 30    # freeze after a catch, so you can see what happened
READY_FRAMES = 45    # freeze before a round, so you can find your square
WINDOW_SIZE = (cfg.ARENA_W * SCALE, cfg.ARENA_H * SCALE)

BACKGROUND = (24, 26, 38)
PLAYER_COLOR = (92, 200, 255)
ENEMY_COLOR = (255, 96, 96)
TEXT_COLOR = (230, 230, 240)

# Physical key positions, so the controls work in any keyboard language.
SCANCODE_ACTIONS = {
    pygame.KSCAN_UP: cfg.UP,
    pygame.KSCAN_W: cfg.UP,
    pygame.KSCAN_DOWN: cfg.DOWN,
    pygame.KSCAN_S: cfg.DOWN,
    pygame.KSCAN_LEFT: cfg.LEFT,
    pygame.KSCAN_A: cfg.LEFT,
    pygame.KSCAN_RIGHT: cfg.RIGHT,
    pygame.KSCAN_D: cfg.RIGHT,
}


def draw_entity(canvas: pygame.Surface, pos: np.ndarray, color: tuple) -> None:
    half = cfg.ENTITY_SIZE // 2
    rect = pygame.Rect(
        int(pos[0]) - half, int(pos[1]) - half, cfg.ENTITY_SIZE, cfg.ENTITY_SIZE
    )
    pygame.draw.rect(canvas, color, rect)


def print_summary(
    round_frames: list[int],
    frame_times: list[int],
    key_downs: int,
    key_ups: int,
    held_at_exit: int,
) -> None:
    """Report the session: how the rounds went, then technical health."""
    print(f"player speed: {cfg.PLAYER_SPEED}, enemy speed: {cfg.ENEMY_SPEED}")
    print(f"rounds: {len(round_frames)}")
    if round_frames:
        seconds = [frames / FPS for frames in round_frames]
        print("round times (s): " + ", ".join(f"{s:.1f}" for s in seconds))
        print(f"median: {np.median(seconds):.1f} s, best: {max(seconds):.1f} s")

    # Skip the first frames, which include window start-up.
    times = np.array(frame_times[10:] or frame_times)
    if len(times):
        print(f"frames: {len(times)}")
        print(f"mean frame: {times.mean():.1f} ms, worst frame: {times.max()} ms")
        print(f"slow frames (over 25 ms): {int((times > 25).sum())}")
    print(f"key downs: {key_downs}, key ups: {key_ups}, held at exit: {held_at_exit}")


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

    held_keys: list[int] = []     # movement keys held down, oldest first
    round_frames: list[int] = []  # length of every finished round, in frames
    frames_alive = 0
    pause_left = 0                # frames left of the freeze after a catch
    ready_left = READY_FRAMES     # frames left of the freeze before a round

    # Technical measurements, printed when the game closes.
    frame_times: list[int] = []
    key_downs = 0
    key_ups = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.scancode in SCANCODE_ACTIONS:
                    key_downs += 1
                    if event.scancode not in held_keys:
                        held_keys.append(event.scancode)
            elif event.type == pygame.KEYUP:
                if event.scancode in SCANCODE_ACTIONS:
                    key_ups += 1
                if event.scancode in held_keys:
                    held_keys.remove(event.scancode)
            elif event.type == pygame.WINDOWFOCUSLOST:
                # Key releases are missed while the window is in the
                # background, so forget everything that was held.
                held_keys.clear()

        # The most recently pressed key wins.
        player_action = SCANCODE_ACTIONS[held_keys[-1]] if held_keys else cfg.STAY

        if pause_left > 0:
            # Frozen after a catch. Then set up the next round.
            pause_left -= 1
            if pause_left == 0:
                arena.reset()
                frames_alive = 0
                ready_left = READY_FRAMES
        elif ready_left > 0:
            # New positions are on screen, but nobody moves yet.
            ready_left -= 1
        else:
            enemy_action = enemy(enemy_observation(arena))
            caught = arena.step(player_action, enemy_action)
            frames_alive += 1
            if caught:
                round_frames.append(frames_alive)
                pause_left = PAUSE_FRAMES

        canvas.fill(BACKGROUND)
        draw_entity(canvas, arena.player_pos, PLAYER_COLOR)
        draw_entity(canvas, arena.enemy_pos, ENEMY_COLOR)
        window.blit(pygame.transform.scale(canvas, WINDOW_SIZE), (0, 0))

        best = max(round_frames, default=0)
        hud = (
            f"round: {len(round_frames) + 1}   "
            f"survived: {frames_alive / FPS:.1f}s   "
            f"best: {best / FPS:.1f}s"
        )
        if pause_left > 0:
            hud += "   CAUGHT!"
        elif ready_left > 0:
            hud += "   GET READY"
        window.blit(font.render(hud, True, TEXT_COLOR), (10, 8))

        pygame.display.flip()
        frame_times.append(clock.tick(FPS))

        # Hands control back for a moment. Needed later to run in a browser.
        await asyncio.sleep(0)

    pygame.quit()
    print_summary(round_frames, frame_times, key_downs, key_ups, len(held_keys))


if __name__ == "__main__":
    asyncio.run(main())
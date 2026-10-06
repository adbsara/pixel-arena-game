"""Play against the trained enemy. Arrow keys or WASD to move, Esc to quit."""

import asyncio
from pathlib import Path

import numpy as np
import pygame

from pixel_arena.game import Arena, enemy_observation
from pixel_arena.game import config as cfg
from pixel_arena.game.policy import NumpyPolicy
from pixel_arena.ui.sprites import CANVAS_SIZE, Sprites

WEIGHTS_PATH = Path("models/chaser.npz")

SCALE = 3            # each logical pixel becomes a 3x3 block on screen
FPS = 60
PAUSE_FRAMES = 30    # freeze after a hit, so you can see what happened
READY_FRAMES = 45    # freeze before a round, so you can find your character
WINDOW_SIZE = (CANVAS_SIZE[0] * SCALE, CANVAS_SIZE[1] * SCALE)

TEXT_COLOR = (240, 240, 245)
TEXT_BACKING = (24, 26, 38)

# Hit effect: all three fade out over the freeze that follows a hit.
SHAKE_PIXELS = 8               # how far the picture jumps at the start
FLASH_COLOR = (210, 40, 40)
FLASH_ALPHA = 120              # how solid the red flash is at the start
BLINK_FRAMES = 4               # the player is hidden / shown this long

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


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def print_summary(
    round_frames: list[int],
    round_coins: list[int],
    movement: dict[str, int],
    frame_times: list[int],
) -> None:
    """Report the session: rounds, how the player moved, technical health."""
    print(f"player speed: {cfg.PLAYER_SPEED}, enemy speed: {cfg.ENEMY_SPEED}")
    print(f"rounds: {len(round_frames)}")
    if round_frames:
        seconds = [frames / FPS for frames in round_frames]
        print("round times (s): " + ", ".join(f"{s:.1f}" for s in seconds))
        print("round coins: " + ", ".join(str(c) for c in round_coins))
        print(f"time  -> median: {np.median(seconds):.1f} s, best: {max(seconds):.1f} s")
        print(f"coins -> median: {np.median(round_coins):.1f}, best: {max(round_coins)}")

    # What the player did during play, frame by frame.
    active = movement["active"]
    print(f"active frames: {active}")
    print(f"  no key held:             {percent(movement['idle'], active)}")
    print(f"  pushing into a wall:     {percent(movement['wall'], active)}")
    print(f"  moving toward the enemy: {percent(movement['toward'], active)}")
    print(f"  moving away from enemy:  {percent(movement['away'], active)}")

    # Skip the first frames, which include window start-up.
    times = np.array(frame_times[10:] or frame_times)
    if len(times):
        print(f"mean frame: {times.mean():.1f} ms, worst frame: {times.max()} ms")


async def main() -> None:
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pixel Arena")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    # Everything is drawn on a small canvas, then blown up without smoothing.
    # That keeps the pixel art crisp.
    sprites = Sprites()
    canvas = pygame.Surface(CANVAS_SIZE)

    # A red sheet laid over the window when the player is hit.
    flash = pygame.Surface(WINDOW_SIZE)
    flash.fill(FLASH_COLOR)

    arena = Arena(np.random.default_rng())
    enemy = NumpyPolicy.load(WEIGHTS_PATH)
    fx_rng = np.random.default_rng()   # randomness for effects only

    held_keys: list[int] = []     # movement keys held down, oldest first
    round_frames: list[int] = []  # length of every finished round, in frames
    round_coins: list[int] = []   # coins collected in every finished round
    frames_alive = 0
    pause_left = 0                # frames left of the freeze after a hit
    ready_left = READY_FRAMES     # frames left of the freeze before a round
    round_over = False            # the last hit took the final life

    # What the player did on each frame of play.
    movement = {"active": 0, "idle": 0, "wall": 0, "toward": 0, "away": 0}
    frame_times: list[int] = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.scancode in SCANCODE_ACTIONS:
                    if event.scancode not in held_keys:
                        held_keys.append(event.scancode)
            elif event.type == pygame.KEYUP:
                if event.scancode in held_keys:
                    held_keys.remove(event.scancode)
            elif event.type == pygame.WINDOWFOCUSLOST:
                # Key releases are missed while the window is in the
                # background, so forget everything that was held.
                held_keys.clear()

        # The most recently pressed key wins.
        player_action = SCANCODE_ACTIONS[held_keys[-1]] if held_keys else cfg.STAY

        if pause_left > 0:
            # Frozen after a hit.
            pause_left -= 1
            if pause_left == 0:
                if round_over:
                    # Out of lives: set up the next round.
                    arena.reset()
                    frames_alive = 0
                    round_over = False
                    ready_left = READY_FRAMES
                else:
                    # Lives left: push the enemy back and carry on.
                    arena.send_enemy_away()
        elif ready_left > 0:
            # New positions are on screen, but nobody moves yet.
            ready_left -= 1
        else:
            prev_player = arena.player_pos.copy()
            prev_enemy = arena.enemy_pos.copy()

            enemy_action = enemy(enemy_observation(arena))
            hit = arena.step(player_action, enemy_action)
            frames_alive += 1

            # Classify this frame by what the player's own move did.
            movement["active"] += 1
            if player_action == cfg.STAY:
                movement["idle"] += 1
            elif not np.any(arena.player_vel):
                movement["wall"] += 1
            else:
                before = np.linalg.norm(prev_player - prev_enemy)
                after = np.linalg.norm(arena.player_pos - prev_enemy)
                movement["toward" if after < before else "away"] += 1

            if hit:
                pause_left = PAUSE_FRAMES
                if arena.lives == 0:
                    round_over = True
                    round_frames.append(frames_alive)
                    round_coins.append(arena.score)

        # Strength of the hit effect: 1 right after a hit, fading to 0.
        hit_strength = pause_left / PAUSE_FRAMES

        canvas.blit(sprites.background, (0, 0))
        sprites.draw(canvas, sprites.coin, arena.coin_pos)
        # The player blinks while the game is frozen after a hit.
        if pause_left == 0 or (pause_left // BLINK_FRAMES) % 2 == 0:
            sprites.draw(canvas, sprites.player, arena.player_pos)
        sprites.draw(canvas, sprites.enemy, arena.enemy_pos)

        # Screen shake: the whole picture jumps around, less and less.
        reach = int(SHAKE_PIXELS * hit_strength)
        if reach > 0:
            offset = (
                int(fx_rng.integers(-reach, reach + 1)),
                int(fx_rng.integers(-reach, reach + 1)),
            )
        else:
            offset = (0, 0)
        window.fill((0, 0, 0))
        window.blit(pygame.transform.scale(canvas, WINDOW_SIZE), offset)

        # Red flash that fades out.
        if hit_strength > 0:
            flash.set_alpha(int(FLASH_ALPHA * hit_strength))
            window.blit(flash, (0, 0))

        best_coins = max(round_coins, default=0)
        hud = (
            f" round: {len(round_frames) + (0 if round_over else 1)}   "
            f"lives: {arena.lives}   "
            f"potions: {arena.score}   "
            f"best: {best_coins} "
        )
        if pause_left > 0:
            hud += "  ROUND OVER " if round_over else "  HIT! "
        elif ready_left > 0:
            hud += "  GET READY "
        window.blit(font.render(hud, True, TEXT_COLOR, TEXT_BACKING), (10, 10))

        pygame.display.flip()
        frame_times.append(clock.tick(FPS))

        # Hands control back for a moment. Needed later to run in a browser.
        await asyncio.sleep(0)

    pygame.quit()
    print_summary(round_frames, round_coins, movement, frame_times)


if __name__ == "__main__":
    asyncio.run(main())
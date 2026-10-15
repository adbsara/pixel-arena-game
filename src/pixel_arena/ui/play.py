"""Play against the trained enemy.

Arrow keys or WASD to move, Space to swing the sword, Esc to quit.
"""

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

# Sword: a facing line plus a ring, shown for a few frames after a swing.
# The ring is the real hitbox (see ATTACK_RANGE); the line is only for
# feedback, so it points the way the player last moved.
SWORD_COLOR = (230, 230, 230)
SWORD_HIT_COLOR = (255, 220, 90)
SWORD_VISIBLE_FRAMES = 12      # how long the swing stays on screen
SWORD_HIT_FRAMES = 10          # how long the "hit" look lasts, within that
SWING_SHAKE_PIXELS = 3         # a small jolt on every swing, hit or not

STUN_TINT = (140, 140, 255, 90)   # laid over a stunned enemy
STRUCK_COLOR = (255, 230, 120)
STRUCK_FRAMES = 18                # how long the hit spark lasts

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

    active = movement["active"]
    print(f"active frames: {active}")
    print(f"  no key held:             {percent(movement['idle'], active)}")
    print(f"  pushing into a wall:     {percent(movement['wall'], active)}")
    print(f"  moving toward the enemy: {percent(movement['toward'], active)}")
    print(f"  moving away from enemy:  {percent(movement['away'], active)}")

    times = np.array(frame_times[10:] or frame_times)
    if len(times):
        print(f"mean frame: {times.mean():.1f} ms, worst frame: {times.max()} ms")


def draw_sword(canvas: pygame.Surface, arena: Arena) -> None:
    """A facing line plus a ring showing the real, all-around hit range."""
    # attack_cooldown counts down from ATTACK_COOLDOWN right after a swing,
    # so a high value here means the swing just happened.
    frames_since_swing = cfg.ATTACK_COOLDOWN - arena.attack_cooldown
    if not (0 <= frames_since_swing <= SWORD_VISIBLE_FRAMES):
        return

    half = cfg.TILE // 2
    origin = arena.player_pos + np.array([half, half])
    direction = cfg.DIRECTIONS[arena.facing]

    recent_hit = arena.just_struck or frames_since_swing <= SWORD_HIT_FRAMES
    color = SWORD_HIT_COLOR if recent_hit else SWORD_COLOR
    width = 3 if recent_hit else 2

    # The facing line: feedback only, not the real hitbox.
    tip = origin + direction * cfg.ATTACK_RANGE
    pygame.draw.line(canvas, color, origin.astype(int), tip.astype(int), width)

    # The range ring: this circle is the actual hitbox.
    pygame.draw.circle(canvas, color, origin.astype(int), int(cfg.ATTACK_RANGE), width=1)


def draw_stun_tint(canvas: pygame.Surface, arena: Arena) -> None:
    """A translucent overlay on a stunned enemy."""
    if arena.enemy_stun == 0:
        return
    half = cfg.TILE // 2
    overlay = pygame.Surface((cfg.TILE, cfg.TILE), pygame.SRCALPHA)
    overlay.fill(STUN_TINT)
    canvas.blit(overlay, (int(arena.enemy_pos[0]) - half, int(arena.enemy_pos[1]) - half))


def draw_struck_spark(canvas: pygame.Surface, pos: np.ndarray, age: int) -> None:
    """An expanding, fading ring at the spot a hit landed."""
    progress = age / STRUCK_FRAMES
    radius = int(4 + 14 * progress)
    alpha = max(0, int(255 * (1 - progress)))
    half = cfg.TILE // 2
    overlay = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(overlay, (*STRUCK_COLOR, alpha), (radius, radius), radius, width=2)
    canvas.blit(overlay, (int(pos[0]) + half - radius, int(pos[1]) + half - radius))


async def main() -> None:
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pixel Arena")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    sprites = Sprites()
    canvas = pygame.Surface(CANVAS_SIZE)

    flash = pygame.Surface(WINDOW_SIZE)
    flash.fill(FLASH_COLOR)

    arena = Arena(np.random.default_rng())
    enemy = NumpyPolicy.load(WEIGHTS_PATH)
    fx_rng = np.random.default_rng()

    held_keys: list[int] = []
    attack_pressed = False        # True for exactly one frame per key press
    round_frames: list[int] = []
    round_coins: list[int] = []
    frames_alive = 0
    pause_left = 0
    ready_left = READY_FRAMES
    round_over = False

    struck_pos: np.ndarray | None = None   # where the last successful hit landed
    struck_age = 0                         # frames since that hit, counts up
    swing_shake = False   # True for one frame right after a swing

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
                elif event.key == pygame.K_SPACE:
                    attack_pressed = True
                elif event.scancode in SCANCODE_ACTIONS:
                    if event.scancode not in held_keys:
                        held_keys.append(event.scancode)
            elif event.type == pygame.KEYUP:
                if event.scancode in held_keys:
                    held_keys.remove(event.scancode)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held_keys.clear()

        player_action = SCANCODE_ACTIONS[held_keys[-1]] if held_keys else cfg.STAY
        attack = attack_pressed
        attack_pressed = False   # consumed whether or not it lands this frame

        if pause_left > 0:
            pause_left -= 1
            if pause_left == 0:
                if round_over:
                    arena.reset()
                    frames_alive = 0
                    round_over = False
                    ready_left = READY_FRAMES
                else:
                    arena.send_enemy_away()
        elif ready_left > 0:
            ready_left -= 1
        else:
            prev_player = arena.player_pos.copy()
            prev_enemy = arena.enemy_pos.copy()

            enemy_action = enemy(enemy_observation(arena))
            hit = arena.step(player_action, enemy_action, attack=attack)
            frames_alive += 1

            if arena.just_struck:
                struck_pos = arena.enemy_pos.copy()
                struck_age = 0
            if arena.just_attacked:
                swing_shake = True

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

        hit_strength = pause_left / PAUSE_FRAMES

        if struck_pos is not None:
            struck_age += 1
            if struck_age > STRUCK_FRAMES:
                struck_pos = None

        canvas.blit(sprites.background, (0, 0))
        sprites.draw(canvas, sprites.coin, arena.coin_pos)
        draw_sword(canvas, arena)
        if pause_left == 0 or (pause_left // BLINK_FRAMES) % 2 == 0:
            sprites.draw(canvas, sprites.player, arena.player_pos)
        sprites.draw(canvas, sprites.enemy, arena.enemy_pos)
        draw_stun_tint(canvas, arena)
        if struck_pos is not None:
            draw_struck_spark(canvas, struck_pos, struck_age)

        reach = int(SHAKE_PIXELS * hit_strength)
        if swing_shake:
            reach = max(reach, SWING_SHAKE_PIXELS)
            swing_shake = False
        if reach > 0:
            offset = (
                int(fx_rng.integers(-reach, reach + 1)),
                int(fx_rng.integers(-reach, reach + 1)),
            )
        else:
            offset = (0, 0)
        window.fill((0, 0, 0))
        window.blit(pygame.transform.scale(canvas, WINDOW_SIZE), offset)

        if hit_strength > 0:
            flash.set_alpha(int(FLASH_ALPHA * hit_strength))
            window.blit(flash, (0, 0))

        best_coins = max(round_coins, default=0)
        hud = (
            f" round: {len(round_frames) + (0 if round_over else 1)}   "
            f"lives: {arena.lives}   "
            f"potions: {arena.score}   "
            f"swords: {arena.swords}   "
            f"best: {best_coins} "
        )
        if pause_left > 0:
            hud += "  ROUND OVER " if round_over else "  HIT! "
        elif ready_left > 0:
            hud += "  GET READY "
        window.blit(font.render(hud, True, TEXT_COLOR, TEXT_BACKING), (10, 10))

        pygame.display.flip()
        frame_times.append(clock.tick(FPS))

        await asyncio.sleep(0)

    pygame.quit()
    print_summary(round_frames, round_coins, movement, frame_times)


if __name__ == "__main__":
    asyncio.run(main())
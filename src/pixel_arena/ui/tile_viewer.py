"""Show every tile of the art pack with its index number. Esc to quit."""

from pathlib import Path

import pygame

from pixel_arena.game import config as cfg

SHEET_PATH = Path("assets/tiny_dungeon/tilemap_packed.png")
OUTPUT_PATH = Path("assets/tile_index.png")

ZOOM = 3          # each tile is drawn three times its real size
LABEL_H = 16      # room under each tile for its number
PAD = 8           # gap between columns
BACKGROUND = (60, 62, 74)
TEXT_COLOR = (255, 255, 255)


def main() -> None:
    pygame.init()

    sheet = pygame.image.load(SHEET_PATH)
    cols = sheet.get_width() // cfg.TILE
    rows = sheet.get_height() // cfg.TILE

    cell_w = cfg.TILE * ZOOM + PAD
    cell_h = cfg.TILE * ZOOM + LABEL_H
    window = pygame.display.set_mode((cols * cell_w, rows * cell_h))
    pygame.display.set_caption("Tile viewer")
    font = pygame.font.Font(None, 18)
    sheet = sheet.convert_alpha()

    window.fill(BACKGROUND)
    for index in range(cols * rows):
        col, row = index % cols, index // cols

        # Cut one tile out of the sheet and enlarge it without smoothing.
        tile = sheet.subsurface(
            (col * cfg.TILE, row * cfg.TILE, cfg.TILE, cfg.TILE)
        )
        tile = pygame.transform.scale(tile, (cfg.TILE * ZOOM, cfg.TILE * ZOOM))

        x = col * cell_w + PAD // 2
        y = row * cell_h
        window.blit(tile, (x, y))
        window.blit(font.render(str(index), True, TEXT_COLOR), (x, y + cfg.TILE * ZOOM + 1))

    pygame.display.flip()
    pygame.image.save(window, OUTPUT_PATH)
    print(f"{cols} columns x {rows} rows = {cols * rows} tiles")
    print(f"numbered sheet saved to {OUTPUT_PATH}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        pygame.time.wait(50)

    pygame.quit()


if __name__ == "__main__":
    main()
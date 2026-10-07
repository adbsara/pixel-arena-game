"""Assemble a self-contained folder that pygbag can turn into a web page."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web"

# The entry point of the web version. The web runtime needs numpy to be
# imported at the very top of this file, before anything else that uses it.
MAIN = '''\
# /// script
# dependencies = [
#   "numpy",
#   "pygame-ce",
# ]
# ///
import asyncio

import numpy
import pygame

from pixel_arena.ui.play import main

asyncio.run(main())
'''

SKIP = shutil.ignore_patterns("__pycache__")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    # Only what the game needs to run: no training code, no PyTorch.
    package = OUT / "pixel_arena"
    source = ROOT / "src" / "pixel_arena"
    shutil.copytree(source / "game", package / "game", ignore=SKIP)
    shutil.copytree(source / "ui", package / "ui", ignore=SKIP)
    shutil.copy(source / "__init__.py", package / "__init__.py")

    art = OUT / "assets" / "tiny_dungeon"
    art.mkdir(parents=True)
    shutil.copy(ROOT / "assets" / "tiny_dungeon" / "tilemap_packed.png", art)

    models = OUT / "models"
    models.mkdir()
    shutil.copy(ROOT / "models" / "chaser.npz", models)

    (OUT / "main.py").write_text(MAIN, encoding="utf-8")
    print(f"web folder ready: {OUT}")


if __name__ == "__main__":
    main()
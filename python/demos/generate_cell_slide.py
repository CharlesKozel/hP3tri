"""Generate a presentation slide showing all cell types with hex icons and names.

Usage:
    .venv/bin/python python/demos/generate_cell_slide.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from simulator.cell_types import CellType, CELL_PROPERTIES

OUTPUT = Path(__file__).parent / "output" / "cell_types_slide.png"

W, H = 1920, 1080
BG = (6, 6, 14)

CELL_TYPES_TO_SHOW = [
    CellType.PHOTOSYNTHETIC,
    CellType.SOFT_TISSUE,
    CellType.MOUTH,
    CellType.FLAGELLA,
    CellType.EYE,
    CellType.SPIKE,
    CellType.ARMOR,
    CellType.SKIN,
    CellType.FOOD,
]


def hex_color_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip('#')
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def hex_corners(
    cx: float, cy: float, size: float,
) -> list[tuple[float, float]]:
    return [
        (cx + size * math.cos(math.radians(60 * i)),
         cy + size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default(size=size)


def draw_hex_cell(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    size: float,
    color: tuple[int, int, int],
    cell_type: CellType,
) -> None:
    corners = hex_corners(cx, cy, size)
    draw.polygon(corners, fill=color)
    draw.polygon(corners, outline=(3, 3, 8), width=3)

    if cell_type == CellType.EYE:
        pr = size * 0.28
        draw.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(10, 10, 30))


def make_glow_layer(
    positions: list[tuple[float, float, tuple[int, int, int]]],
) -> Image.Image:
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for cx, cy, color in positions:
        scaled = tuple(min(255, int(c * 0.5)) for c in color)
        corners = hex_corners(cx, cy, 80)
        gd.polygon(corners, fill=scaled)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=40))
    return glow


def generate() -> Path:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    title_font = load_font(48, bold=True)
    label_font = load_font(24)

    draw.text(
        (W // 2, 60), "Cell Types",
        fill=(220, 235, 250), font=title_font, anchor="mm",
    )

    n = len(CELL_TYPES_TO_SHOW)
    hex_size = 60
    spacing_x = 192
    total_w = (n - 1) * spacing_x
    start_x = (W - total_w) / 2
    center_y = H // 2 - 20

    glow_positions: list[tuple[float, float, tuple[int, int, int]]] = []
    for i, ct in enumerate(CELL_TYPES_TO_SHOW):
        props = CELL_PROPERTIES[ct]
        color = hex_color_to_rgb(props.color)
        cx = start_x + i * spacing_x
        glow_positions.append((cx, center_y, color))

    glow = make_glow_layer(glow_positions)
    img = ImageChops.add(img, glow)
    draw = ImageDraw.Draw(img)

    draw.text(
        (W // 2, 60), "Cell Types",
        fill=(220, 235, 250), font=title_font, anchor="mm",
    )

    for i, ct in enumerate(CELL_TYPES_TO_SHOW):
        props = CELL_PROPERTIES[ct]
        color = hex_color_to_rgb(props.color)
        cx = start_x + i * spacing_x
        cy = center_y

        draw_hex_cell(draw, cx, cy, hex_size, color, ct)

        draw.text(
            (cx, cy + hex_size + 30), props.display_name,
            fill=(200, 210, 225), font=label_font, anchor="mm",
        )

    ly = H - 120
    draw.line([(W // 2 - 300, ly), (W // 2 + 300, ly)], fill=(30, 40, 55), width=2)
    subtitle_font = load_font(20)
    draw.text(
        (W // 2, ly + 30), "hP3tri — Artificial Life Evolution Simulator",
        fill=(60, 75, 95), font=subtitle_font, anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT, quality=95)
    print(f"Slide saved: {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
    return OUTPUT


if __name__ == "__main__":
    generate()

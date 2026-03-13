"""Generate logo for hP3tri with an evolved organism.

Usage:
    .venv/bin/python python/demos/generate_logo.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from simulator.cell_types import CellType

OUTPUT = Path(__file__).parent / "output" / "logo.png"

W, H = 2000, 2400
ORG_HEX = 44
BG = (6, 6, 14)

COLORS: dict[CellType, tuple[int, int, int]] = {
    CellType.PHOTOSYNTHETIC: (51, 170, 51),
    CellType.SOFT_TISSUE:    (232, 180, 160),
    CellType.MOUTH:          (204, 51, 51),
    CellType.FLAGELLA:       (170, 110, 200),
    CellType.EYE:            (240, 245, 255),
    CellType.SPIKE:          (255, 102, 0),
    CellType.ARMOR:          (120, 120, 155),
    CellType.SKIN:           (200, 170, 120),
}

GLOW_TINT: dict[CellType, tuple[int, int, int]] = {
    CellType.PHOTOSYNTHETIC: (30, 200, 60),
    CellType.MOUTH:          (255, 40, 30),
    CellType.EYE:            (180, 220, 255),
    CellType.SPIKE:          (255, 130, 20),
    CellType.FLAGELLA:       (180, 100, 255),
}

# Bilateral predator-hunter organism (mirror axis: y=0 via (q,r)->(q,-q-r))
ORGANISM: dict[tuple[int, int], CellType] = {
    # ── Center axis (q + 2r = 0) ──
    (0, 0): CellType.PHOTOSYNTHETIC,
    (2, -1): CellType.MOUTH,
    (-2, 1): CellType.ARMOR,
    (4, -2): CellType.SPIKE,

    # ── Top half ──
    (-1, 0): CellType.PHOTOSYNTHETIC,
    (0, -1): CellType.PHOTOSYNTHETIC,
    (1, -1): CellType.MOUTH,
    (0, -2): CellType.FLAGELLA,
    (-1, -1): CellType.FLAGELLA,
    (1, -2): CellType.SKIN,
    (2, -2): CellType.EYE,
    (-2, 0): CellType.SKIN,
    (-3, 0): CellType.SPIKE,
    (3, -1): CellType.SPIKE,
    (-1, -2): CellType.SKIN,
    (-2, -1): CellType.SPIKE,

    # ── Bottom half (mirror: (q, -q-r)) ──
    (-1, 1): CellType.PHOTOSYNTHETIC,
    (0, 1): CellType.PHOTOSYNTHETIC,
    (1, 0): CellType.MOUTH,
    (0, 2): CellType.FLAGELLA,
    (-1, 2): CellType.FLAGELLA,
    (1, 1): CellType.SKIN,
    (2, 0): CellType.EYE,
    (-2, 2): CellType.SKIN,
    (-3, 2): CellType.SPIKE,
    (3, -2): CellType.SPIKE,
    (-1, 3): CellType.SKIN,
    (-2, 3): CellType.SPIKE,
}


def hex_corners(
    cx: float, cy: float, size: float,
) -> list[tuple[float, float]]:
    return [
        (cx + size * math.cos(math.radians(60 * i)),
         cy + size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def axial_to_pixel(q: int, r: int, size: float) -> tuple[float, float]:
    x = size * (3.0 / 2.0 * q)
    y = size * (math.sqrt(3) / 2.0 * q + math.sqrt(3) * r)
    return x, y


def draw_bg_grid(draw: ImageDraw.ImageDraw) -> None:
    """Subtle dot grid at hex centers across the canvas."""
    s = 16
    cols = int(W / (s * 1.5)) + 4
    rows = int(H / (s * math.sqrt(3))) + 4
    for q in range(-2, cols):
        for r in range(-2, rows):
            px, py = axial_to_pixel(q, r, s)
            if -s < px < W + s and -s < py < H + s:
                draw.ellipse(
                    [px - 1.2, py - 1.2, px + 1.2, py + 1.2],
                    fill=(18, 18, 32),
                )


def draw_organism(
    draw: ImageDraw.ImageDraw, cx: int, cy: int,
    outline: bool = True,
) -> None:
    for (q, r), ct in ORGANISM.items():
        px, py = axial_to_pixel(q, r, ORG_HEX)
        hx, hy = cx + px, cy + py
        corners = hex_corners(hx, hy, ORG_HEX * 0.92)
        color = COLORS[ct]
        draw.polygon(corners, fill=color)
        if outline:
            draw.polygon(corners, outline=(3, 3, 8), width=2)

    # Pupils on eyes
    for (q, r), ct in ORGANISM.items():
        if ct != CellType.EYE:
            continue
        px, py = axial_to_pixel(q, r, ORG_HEX)
        ex, ey = cx + px, cy + py
        pr = ORG_HEX * 0.22
        draw.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=(10, 10, 30))


def make_glow(cx: int, cy: int) -> Image.Image:
    """Create a colored glow layer behind the organism."""
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)

    for (q, r), ct in ORGANISM.items():
        px, py = axial_to_pixel(q, r, ORG_HEX)
        hx, hy = cx + px, cy + py
        tint = GLOW_TINT.get(ct, COLORS.get(ct, (40, 80, 40)))
        scaled = tuple(min(255, int(c * 0.7)) for c in tint)
        corners = hex_corners(hx, hy, ORG_HEX * 1.3)
        gd.polygon(corners, fill=scaled)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=50))
    return glow


def make_ambient_glow(cx: int, cy: int) -> Image.Image:
    """Large soft radial glow centered on organism."""
    ambient = Image.new("RGB", (W, H), (0, 0, 0))
    ad = ImageDraw.Draw(ambient)
    radius = 500
    for i in range(radius, 0, -2):
        frac = i / radius
        intensity = int(18 * (1 - frac) ** 2)
        color = (0, intensity, int(intensity * 0.7))
        ad.ellipse(
            [cx - i, cy - i, cx + i, cy + i],
            fill=color,
        )
    return ambient


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


def draw_title(draw: ImageDraw.ImageDraw) -> None:
    title_font = load_font(160, bold=True)
    sub_font = load_font(52)

    title = "hP3tri"
    ty = H - 480

    # Title shadow
    draw.text((W // 2 + 3, ty + 3), title, fill=(0, 0, 0), font=title_font, anchor="mm")
    # Title
    draw.text((W // 2, ty), title, fill=(220, 235, 250), font=title_font, anchor="mm")

    # Subtitle
    sub = "Artificial Life Evolution Simulator"
    draw.text(
        (W // 2, ty + 120), sub,
        fill=(80, 100, 120), font=sub_font, anchor="mm",
    )

    # Decorative line
    lw = 400
    ly = ty + 180
    draw.line([(W // 2 - lw, ly), (W // 2 + lw, ly)], fill=(30, 40, 55), width=2)


def generate() -> Path:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Background dot grid
    draw_bg_grid(draw)

    org_cx, org_cy = W // 2 - 30, H // 2 - 280

    # Ambient glow
    ambient = make_ambient_glow(org_cx, org_cy)
    img = ImageChops.add(img, ambient)

    # Cell glow
    glow = make_glow(org_cx, org_cy)
    img = ImageChops.add(img, glow)

    # Sharp organism on top
    draw = ImageDraw.Draw(img)
    draw_organism(draw, org_cx, org_cy)

    # Title and subtitle
    draw_title(draw)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT, quality=95)
    print(f"Logo saved: {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")

    # Also save a smaller icon version (1024x1024, organism only)
    icon = Image.new("RGB", (1024, 1024), BG)
    icon_draw = ImageDraw.Draw(icon)

    # Scale down: draw at center of 1024x1024
    icon_cx, icon_cy = 480, 512

    # Ambient glow for icon
    icon_ambient = Image.new("RGB", (1024, 1024), (0, 0, 0))
    iad = ImageDraw.Draw(icon_ambient)
    for i in range(350, 0, -2):
        frac = i / 350
        intensity = int(15 * (1 - frac) ** 2)
        iad.ellipse(
            [icon_cx - i, icon_cy - i, icon_cx + i, icon_cy + i],
            fill=(0, intensity, int(intensity * 0.7)),
        )
    icon = ImageChops.add(icon, icon_ambient)

    # Draw icon grid
    icon_draw = ImageDraw.Draw(icon)
    s = 10
    icols = int(1024 / (s * 1.5)) + 4
    irows = int(1024 / (s * math.sqrt(3))) + 4
    for q in range(-2, icols):
        for r in range(-2, irows):
            px, py = axial_to_pixel(q, r, s)
            if 0 < px < 1024 and 0 < py < 1024:
                icon_draw.ellipse([px - 0.8, py - 0.8, px + 0.8, py + 0.8], fill=(15, 15, 28))

    # Glow for icon
    icon_hex = 30
    icon_glow = Image.new("RGB", (1024, 1024), (0, 0, 0))
    igd = ImageDraw.Draw(icon_glow)
    for (q, r), ct in ORGANISM.items():
        px, py = axial_to_pixel(q, r, icon_hex)
        hx, hy = icon_cx + px, icon_cy + py
        tint = GLOW_TINT.get(ct, COLORS.get(ct, (40, 80, 40)))
        scaled = tuple(min(255, int(c * 0.6)) for c in tint)
        corners = hex_corners(hx, hy, icon_hex * 1.2)
        igd.polygon(corners, fill=scaled)
    icon_glow = icon_glow.filter(ImageFilter.GaussianBlur(radius=35))
    icon = ImageChops.add(icon, icon_glow)

    icon_draw = ImageDraw.Draw(icon)
    for (q, r), ct in ORGANISM.items():
        px, py = axial_to_pixel(q, r, icon_hex)
        hx, hy = icon_cx + px, icon_cy + py
        corners = hex_corners(hx, hy, icon_hex * 0.92)
        icon_draw.polygon(corners, fill=COLORS[ct])
        icon_draw.polygon(corners, outline=(3, 3, 8), width=2)
    for (q, r), ct in ORGANISM.items():
        if ct == CellType.EYE:
            px, py = axial_to_pixel(q, r, icon_hex)
            ex, ey = icon_cx + px, icon_cy + py
            pr = icon_hex * 0.2
            icon_draw.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=(10, 10, 30))

    icon_path = OUTPUT.parent / "logo_icon.png"
    icon.save(icon_path, quality=95)
    print(f"Icon saved: {icon_path} ({icon_path.stat().st_size // 1024} KB)")

    return OUTPUT


if __name__ == "__main__":
    generate()

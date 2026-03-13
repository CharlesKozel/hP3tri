"""Render demo replay JSON files to PNG frames using Pillow.

Usage:
    .venv/bin/python python/demos/render_frames.py [scene_name]
    .venv/bin/python python/demos/render_frames.py          # all scenes
    .venv/bin/python python/demos/render_frames.py genesis   # single scene

Outputs PNGs to python/demos/output/{scene_name}/ and prints an ffmpeg command.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow required: .venv/bin/pip install Pillow")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.cell_types import CellType, CELL_PROPERTIES

OUTPUT_DIR = Path(__file__).parent / "output"

BACKGROUND = (17, 17, 17)

GENOME_TINTS: dict[int, tuple[int, int, int]] = {
    1: (255, 102, 68),
    2: (204, 68, 255),
    3: (68, 136, 255),
    4: (68, 204, 68),
    5: (255, 204, 34),
    6: (255, 68, 170),
    7: (68, 255, 204),
    8: (204, 136, 68),
    9: (136, 68, 255),
    10: (68, 255, 102),
}


def hex_color_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip('#')
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def blend(base: tuple[int, int, int], tint: tuple[int, int, int], alpha: float = 0.3) -> tuple[int, int, int]:
    return tuple(int(b * (1 - alpha) + t * alpha) for b, t in zip(base, tint))  # type: ignore


def flat_top_hex_corners(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    """Flat-top hexagon vertex coordinates."""
    corners = []
    for i in range(6):
        angle = math.pi / 180 * (60 * i)
        corners.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    return corners


def axial_to_pixel(q: int, r: int, hex_size: float) -> tuple[float, float]:
    """Convert axial hex coords to pixel center (flat-top)."""
    x = hex_size * (3 / 2 * q)
    y = hex_size * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return x, y


def render_frame(
    frame: dict,
    hex_size: float = 6.0,
    padding: int = 10,
) -> Image.Image:
    grid = frame["grid"]
    width = grid["width"]
    height = grid["height"]
    tiles = grid["tiles"]

    # Compute image dimensions from grid bounds
    max_px, max_py = axial_to_pixel(width - 1, height - 1, hex_size)
    img_w = int(max_px + hex_size * 2 + padding * 2)
    img_h = int(max_py + hex_size * 2 + padding * 2)

    img = Image.new("RGB", (img_w, img_h), BACKGROUND)
    draw = ImageDraw.Draw(img)

    for tile in tiles:
        q, r = tile["q"], tile["r"]
        ct = tile["cellType"]
        oid = tile.get("organismId", 0)

        props = CELL_PROPERTIES.get(CellType(ct))
        if props is None:
            continue

        base_rgb = hex_color_to_rgb(props.color)

        # Apply genome tint
        if oid > 0:
            genome_id = 0
            for org in frame.get("organisms", []):
                if org["id"] == oid:
                    genome_id = org["genomeId"]
                    break
            tint = GENOME_TINTS.get(genome_id)
            if tint:
                color = blend(base_rgb, tint, 0.25)
            else:
                color = base_rgb
        else:
            color = base_rgb

        px, py = axial_to_pixel(q, r, hex_size)
        cx = px + padding + hex_size
        cy = py + padding + hex_size

        corners = flat_top_hex_corners(cx, cy, hex_size * 0.9)
        draw.polygon(corners, fill=color)

    return img


def render_scene(scene_name: str) -> None:
    json_path = OUTPUT_DIR / f"{scene_name}.json"
    if not json_path.exists():
        print(f"  Missing {json_path} — run run_demos.py first")
        return

    print(f"\nRendering {scene_name}...")
    with open(json_path) as f:
        data = json.load(f)

    frames = data["frames"]
    scene_dir = OUTPUT_DIR / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        img = render_frame(frame)
        out_path = scene_dir / f"frame_{i:04d}.png"
        img.save(out_path)
        if i % 20 == 0:
            print(f"  frame {i}/{len(frames)}")

    print(f"  {len(frames)} PNGs saved to {scene_dir}/")
    print(f"  ffmpeg command:")
    print(f"    ffmpeg -framerate 15 -i {scene_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {OUTPUT_DIR}/{scene_name}.mp4")


def main() -> None:
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None

    json_files = sorted(OUTPUT_DIR.glob("*.json"))
    if not json_files:
        print("No replay JSON files found. Run run_demos.py first.")
        sys.exit(1)

    scene_names = [f.stem for f in json_files]
    if filter_name:
        if filter_name not in scene_names:
            print(f"Unknown scene: {filter_name}")
            print(f"Available: {', '.join(scene_names)}")
            sys.exit(1)
        scene_names = [filter_name]

    for name in scene_names:
        render_scene(name)

    print(f"\nDone! All frames in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

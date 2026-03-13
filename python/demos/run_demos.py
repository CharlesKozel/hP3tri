"""Run all demo scenes and output replay JSON files.

Usage:
    .venv/bin/python python/demos/run_demos.py [scene_name]
    .venv/bin/python python/demos/run_demos.py          # all scenes
    .venv/bin/python python/demos/run_demos.py genesis   # single scene
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Add project python root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.engine import SimulationEngine
from simulator.cell_types import CellType
from simulator.hex_grid import NEIGHBOR_OFFSETS
from interfaces.brain import BrainProvider

from demos.organisms.templates import (
    Template,
    PLANT_TEMPLATE,
    PREDATOR_TEMPLATE,
    PREY_TEMPLATE,
    WARRIOR_TEMPLATE,
    REPRODUCER_TEMPLATE,
)
from demos.organisms.scripted_brains import (
    PlantBrain,
    PredatorBrain,
    PreyBrain,
    WarriorBrain,
    ReproducerBrain,
)

OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class SceneConfig:
    name: str
    width: int
    height: int
    tick_limit: int
    snapshot_interval: int
    setup: Callable[[SimulationEngine], None]


def prebuild_organism(
    engine: SimulationEngine,
    template: Template,
    center_q: int,
    center_r: int,
    energy: int,
    genome_id: int,
    brain: BrainProvider,
    seed_cell_type: int | None = None,
) -> int:
    """Create an organism and place all template cells at once."""
    if seed_cell_type is None:
        seed_cell_type = template.get((0, 0), int(CellType.SOFT_TISSUE))

    org_id = engine.create_organism(
        seed_q=center_q,
        seed_r=center_r,
        seed_cell_type=seed_cell_type,
        starting_energy=energy,
        genome_id=genome_id,
        brain=brain,
    )

    for (dq, dr), ct in template.items():
        if dq == 0 and dr == 0:
            continue
        q = (center_q + dq) % engine.width
        r = (center_r + dr) % engine.height
        engine.place_cell(org_id, q, r, ct)

    return org_id


def place_food(engine: SimulationEngine, q: int, r: int) -> None:
    idx = r * engine.width + q
    engine.grid[idx].cell_type = int(CellType.FOOD)


def scatter_food(engine: SimulationEngine, count: int, seed: int = 123) -> None:
    """Place food cells at pseudo-random positions, avoiding occupied tiles."""
    import numpy as np
    rng = np.random.default_rng(seed)
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 10:
        q = int(rng.integers(0, engine.width))
        r = int(rng.integers(0, engine.height))
        idx = r * engine.width + q
        if int(engine.grid[idx].cell_type) == int(CellType.NULL):
            place_food(engine, q, r)
            placed += 1
        attempts += 1


# ── Scene Setup Functions ──────────────────────────────────────────


def setup_genesis(engine: SimulationEngine) -> None:
    """Scene 1: Single plant growing from seed."""
    cx, cy = engine.width // 2, engine.height // 2
    engine.create_organism(
        seed_q=cx,
        seed_r=cy,
        seed_cell_type=int(CellType.PHOTOSYNTHETIC),
        starting_energy=999999,
        genome_id=1,
        brain=PlantBrain(),
    )
    engine.genome_registry[1] = {"origin_q": cx, "origin_r": cy}


def setup_hunt(engine: SimulationEngine) -> None:
    """Scene 2: Predator chases prey.

    Predator placed west, seeks east. No food between them to avoid
    blocking movement (occupied tiles block organism movement).
    """
    prebuild_organism(
        engine, PREDATOR_TEMPLATE,
        center_q=15, center_r=32,
        energy=50000, genome_id=1,
        brain=PredatorBrain(seek_direction=0),
    )
    prebuild_organism(
        engine, PREY_TEMPLATE,
        center_q=48, center_r=32,
        energy=30000, genome_id=2,
        brain=PreyBrain(),
    )
    # Food only far from the chase line (above/below)
    for i in range(20):
        q = 30 + (i % 10)
        r = 20 + (i // 10)
        if q < engine.width and r < engine.height:
            place_food(engine, q, r)
    for i in range(20):
        q = 30 + (i % 10)
        r = 44 + (i // 10)
        if q < engine.width and r < engine.height:
            place_food(engine, q, r)


def setup_spike_battle(engine: SimulationEngine) -> None:
    """Scene 3: Two spike warriors charge each other."""
    # Warrior 1 seeks east (dir=0), warrior 2 seeks west (dir=3)
    prebuild_organism(
        engine, WARRIOR_TEMPLATE,
        center_q=14, center_r=24,
        energy=50000, genome_id=1,
        brain=WarriorBrain(seek_direction=0),
    )
    prebuild_organism(
        engine, WARRIOR_TEMPLATE,
        center_q=34, center_r=24,
        energy=50000, genome_id=2,
        brain=WarriorBrain(seek_direction=3),
    )


def setup_lifecycle(engine: SimulationEngine) -> None:
    """Scene 4: Organism grows from seed and reproduces."""
    cx, cy = engine.width // 2, engine.height // 2
    engine.create_organism(
        seed_q=cx,
        seed_r=cy,
        seed_cell_type=int(CellType.PHOTOSYNTHETIC),
        starting_energy=5000,
        genome_id=1,
        brain=ReproducerBrain(growth_ticks=40, reproduce_threshold=500),
    )
    engine.genome_registry[1] = {"origin_q": cx, "origin_r": cy}
    scatter_food(engine, 40)


def setup_ecosystem(engine: SimulationEngine) -> None:
    """Scene 5: Multi-species food chain."""
    plant_brain = PlantBrain()
    prey_brain = PreyBrain()
    predator_brain = PredatorBrain(seek_direction=2)  # NW toward prey cluster

    # 4 plants spread around
    plant_positions = [(20, 20), (70, 20), (20, 70), (70, 70)]
    for pq, pr in plant_positions:
        prebuild_organism(
            engine, PLANT_TEMPLATE,
            center_q=pq, center_r=pr,
            energy=99999, genome_id=1,
            brain=plant_brain,
        )

    # 3 herbivores
    prey_positions = [(30, 45), (60, 45), (48, 30)]
    for pq, pr in prey_positions:
        prebuild_organism(
            engine, PREY_TEMPLATE,
            center_q=pq, center_r=pr,
            energy=30000, genome_id=2,
            brain=prey_brain,
        )

    # 1 predator
    prebuild_organism(
        engine, PREDATOR_TEMPLATE,
        center_q=48, center_r=48,
        energy=50000, genome_id=3,
        brain=predator_brain,
    )

    scatter_food(engine, 50)


# ── Scene Definitions ──────────────────────────────────────────────

SCENES: list[SceneConfig] = [
    SceneConfig("genesis", 48, 48, 150, 2, setup_genesis),
    SceneConfig("hunt", 64, 64, 200, 2, setup_hunt),
    SceneConfig("spike_battle", 48, 48, 150, 2, setup_spike_battle),
    SceneConfig("lifecycle", 64, 64, 250, 3, setup_lifecycle),
    SceneConfig("ecosystem", 96, 96, 300, 3, setup_ecosystem),
]

SCENE_MAP: dict[str, SceneConfig] = {s.name: s for s in SCENES}


def run_scene(scene: SceneConfig) -> Path:
    print(f"\n{'='*60}")
    print(f"  Scene: {scene.name}")
    print(f"  Grid: {scene.width}x{scene.height}, Ticks: {scene.tick_limit}, Interval: {scene.snapshot_interval}")
    print(f"{'='*60}")

    engine = SimulationEngine(scene.width, scene.height, seed=42)
    engine.use_gpu_brain = False

    scene.setup(engine)

    frames: list[dict] = [engine.snapshot()]
    t0 = time.perf_counter()

    for tick in range(1, scene.tick_limit + 1):
        engine.step()
        if tick % scene.snapshot_interval == 0:
            frames.append(engine.snapshot())

        if tick % 50 == 0:
            elapsed = time.perf_counter() - t0
            alive = sum(1 for f in frames[-1]["organisms"] if f["alive"])
            print(f"  tick {tick}/{scene.tick_limit} ({elapsed:.1f}s) — {alive} organisms alive")

    # Always capture final frame
    if scene.tick_limit % scene.snapshot_interval != 0:
        frames.append(engine.snapshot())

    elapsed = time.perf_counter() - t0
    print(f"  Done: {len(frames)} frames in {elapsed:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{scene.name}.json"
    with open(out_path, "w") as f:
        json.dump({"frames": frames}, f)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    return out_path


def main() -> None:
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None

    if filter_name:
        if filter_name not in SCENE_MAP:
            print(f"Unknown scene: {filter_name}")
            print(f"Available: {', '.join(SCENE_MAP.keys())}")
            sys.exit(1)
        scenes = [SCENE_MAP[filter_name]]
    else:
        scenes = SCENES

    print(f"Running {len(scenes)} demo scene(s)...")
    paths: list[Path] = []
    for scene in scenes:
        paths.append(run_scene(scene))

    print(f"\n{'='*60}")
    print("All done! Output files:")
    for p in paths:
        print(f"  {p}")
    print(f"\nView in browser: open MatchViewer and use 'Load File' button")


if __name__ == "__main__":
    main()

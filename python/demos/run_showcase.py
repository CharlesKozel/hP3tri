"""Extended showcase: 100 demo scenes across 20 concepts (5 variations each).

Usage:
    .venv/bin/python python/demos/run_showcase.py              # all 100 scenes
    .venv/bin/python python/demos/run_showcase.py growth        # one concept (5 scenes)
    .venv/bin/python python/demos/run_showcase.py growth_1      # single scene
    .venv/bin/python python/demos/run_showcase.py --render      # simulate + render + compile
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.engine import SimulationEngine
from simulator.cell_types import CellType
from interfaces.brain import BrainProvider

from demos.run_demos import (
    prebuild_organism, place_food, scatter_food, SceneConfig, OUTPUT_DIR,
)
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

# Shorthand cell types
CT = CellType
PHOTO = int(CT.PHOTOSYNTHETIC)
SOFT = int(CT.SOFT_TISSUE)


# ── Parametric Scene Builder ─────────────────────────────────────────

def _setup(
    built: list[tuple[Template, BrainProvider, int, int, int, int]] | None = None,
    seeds: list[tuple[int, BrainProvider, int, int, int, int]] | None = None,
    food: int = 0,
    food_seed: int = 123,
) -> Callable[[SimulationEngine], None]:
    """Factory: returns a setup function from organism placements."""
    def setup(engine: SimulationEngine) -> None:
        for tmpl, brain, q, r, energy, gid in (built or []):
            prebuild_organism(engine, tmpl, q, r, energy, gid, brain)
        for ct, brain, q, r, energy, gid in (seeds or []):
            engine.create_organism(
                seed_q=q, seed_r=r, seed_cell_type=ct,
                starting_energy=energy, genome_id=gid, brain=brain,
            )
        if food:
            scatter_food(engine, food, food_seed)
    return setup


def _food_block(
    engine: SimulationEngine, q0: int, r0: int, cols: int, rows: int,
) -> None:
    for dr in range(rows):
        for dq in range(cols):
            q = (q0 + dq) % engine.width
            r = (r0 + dr) % engine.height
            idx = r * engine.width + q
            if int(engine.grid[idx].cell_type) == int(CT.NULL):
                place_food(engine, q, r)


# ── All 100 Scene Definitions ───────────────────────────────────────

def all_scenes() -> list[SceneConfig]:
    S = SceneConfig
    scenes: list[SceneConfig] = []

    # ────────────────────────────────────────────────────────────
    # 1. GROWTH — organisms blooming from seed
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("growth_1", 48, 48, 150, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 24, 24, 999999, 1),
        ])),
        S("growth_2", 64, 64, 150, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 10, 10, 999999, 1),
            (PHOTO, PlantBrain(), 54, 10, 999999, 2),
            (PHOTO, PlantBrain(), 10, 54, 999999, 3),
            (PHOTO, PlantBrain(), 54, 54, 999999, 4),
        ])),
        S("growth_3", 24, 24, 100, 1, _setup(seeds=[
            (PHOTO, PlantBrain(), 12, 12, 999999, 1),
        ])),
        S("growth_4", 64, 64, 150, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 32, 10, 999999, 1),
            (PHOTO, PlantBrain(), 50, 22, 999999, 2),
            (PHOTO, PlantBrain(), 50, 42, 999999, 3),
            (PHOTO, PlantBrain(), 32, 54, 999999, 4),
            (PHOTO, PlantBrain(), 14, 42, 999999, 5),
            (PHOTO, PlantBrain(), 14, 22, 999999, 6),
        ])),
        S("growth_5", 96, 96, 300, 3, _setup(seeds=[
            (PHOTO, PlantBrain(), 48, 48, 9999999, 1),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 2. HUNT — predator chasing prey
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("hunt_1", 64, 64, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 15, 32, 50000, 1),
            (PREY_TEMPLATE, PreyBrain(), 48, 32, 30000, 2),
        ], food=20, food_seed=200)),
        S("hunt_2", 64, 64, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 25, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 39, 50000, 1),
            (PREY_TEMPLATE, PreyBrain(), 50, 32, 30000, 2),
        ], food=15)),
        S("hunt_3", 48, 48, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 24, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 40, 24, 50000, 1),
            (PREY_TEMPLATE, PreyBrain(), 24, 24, 30000, 2),
        ])),
        S("hunt_4", 128, 64, 300, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 80000, 1),
            (PREY_TEMPLATE, PreyBrain(), 100, 32, 50000, 2),
        ], food=30)),
        S("hunt_5", 64, 64, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 20, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 10, 44, 50000, 3),
            (PREY_TEMPLATE, PreyBrain(), 50, 20, 30000, 2),
            (PREY_TEMPLATE, PreyBrain(), 50, 44, 30000, 4),
        ], food=20)),
    ]

    # ────────────────────────────────────────────────────────────
    # 3. COMBAT — spike warriors fighting
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("combat_1", 48, 48, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 14, 24, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 34, 24, 50000, 2),
        ])),
        S("combat_2", 64, 64, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=5), 16, 16, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=4), 48, 16, 50000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=1), 16, 48, 50000, 3),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=2), 48, 48, 50000, 4),
        ])),
        S("combat_3", 64, 64, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 10, 32, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 54, 26, 50000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 54, 38, 50000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=4), 34, 14, 50000, 3),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=1), 34, 50, 50000, 3),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=2), 54, 32, 50000, 4),
        ])),
        S("combat_4", 48, 48, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 14, 24, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 34, 24, 50000, 2),
        ])),
        S("combat_5", 64, 64, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 10, 32, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 54, 32, 50000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=4), 32, 10, 50000, 3),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 4. REPRODUCTION — organisms creating offspring
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("repro_1", 64, 64, 250, 3, _setup(seeds=[
            (PHOTO, ReproducerBrain(growth_ticks=40, reproduce_threshold=500), 32, 32, 5000, 1),
        ], food=40)),
        S("repro_2", 64, 64, 250, 3, _setup(seeds=[
            (PHOTO, ReproducerBrain(growth_ticks=30, reproduce_threshold=300), 32, 32, 50000, 1),
        ], food=100)),
        S("repro_3", 64, 64, 300, 3, _setup(seeds=[
            (PHOTO, ReproducerBrain(growth_ticks=40, reproduce_threshold=500), 20, 32, 5000, 1),
            (PHOTO, ReproducerBrain(growth_ticks=40, reproduce_threshold=500), 44, 32, 5000, 2),
        ], food=60)),
        S("repro_4", 48, 48, 200, 2, _setup(seeds=[
            (PHOTO, ReproducerBrain(growth_ticks=20, reproduce_threshold=200), 24, 24, 20000, 1),
        ], food=80)),
        S("repro_5", 64, 64, 300, 3, _setup(
            seeds=[
                (PHOTO, ReproducerBrain(growth_ticks=40, reproduce_threshold=500), 20, 32, 8000, 1),
            ],
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 55, 32, 50000, 2),
            ],
            food=40,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 5. ECOSYSTEM — multi-species food chains
    # ────────────────────────────────────────────────────────────
    def _eco_2(engine: SimulationEngine) -> None:
        for pq, pr in [(20, 20), (70, 70)]:
            prebuild_organism(engine, PLANT_TEMPLATE, pq, pr, 99999, 1, PlantBrain())
        for pq, pr in [(40, 40), (50, 50)]:
            prebuild_organism(engine, PREY_TEMPLATE, pq, pr, 30000, 2, PreyBrain())
        for pq, pr in [(30, 50), (60, 30), (45, 60)]:
            prebuild_organism(engine, PREDATOR_TEMPLATE, pq, pr, 50000, 3,
                              PredatorBrain(seek_direction=0))
        scatter_food(engine, 40)

    def _eco_3(engine: SimulationEngine) -> None:
        for pq, pr in [(20, 30), (70, 30)]:
            prebuild_organism(engine, PLANT_TEMPLATE, pq, pr, 99999, 1, PlantBrain())
        for pq, pr in [(25, 50), (35, 40), (50, 45), (65, 50), (45, 60)]:
            prebuild_organism(engine, PREY_TEMPLATE, pq, pr, 30000, 2, PreyBrain())
        prebuild_organism(engine, PREDATOR_TEMPLATE, 48, 48, 50000, 3,
                          PredatorBrain(seek_direction=2))
        scatter_food(engine, 60)

    scenes += [
        S("eco_1", 96, 96, 300, 3, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 20, 20, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 70, 20, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 20, 70, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 70, 70, 99999, 1),
                (PREY_TEMPLATE, PreyBrain(), 30, 45, 30000, 2),
                (PREY_TEMPLATE, PreyBrain(), 60, 45, 30000, 2),
                (PREY_TEMPLATE, PreyBrain(), 48, 30, 30000, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=2), 48, 48, 50000, 3),
            ],
            food=50,
        )),
        S("eco_2", 96, 96, 300, 3, _eco_2),
        S("eco_3", 96, 96, 300, 3, _eco_3),
        S("eco_4", 48, 48, 200, 2, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 12, 12, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 36, 36, 99999, 1),
                (PREY_TEMPLATE, PreyBrain(), 24, 24, 30000, 2),
                (PREY_TEMPLATE, PreyBrain(), 36, 12, 30000, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 12, 36, 50000, 3),
            ],
            food=30,
        )),
        S("eco_5", 96, 96, 300, 3, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 20, 20, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 50, 20, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 80, 20, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 20, 50, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 50, 50, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 80, 50, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 20, 80, 99999, 1),
                (PLANT_TEMPLATE, PlantBrain(), 50, 80, 99999, 1),
                (PREY_TEMPLATE, PreyBrain(), 35, 35, 30000, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 65, 65, 50000, 3),
            ],
            food=40,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 6. SWARM — many organisms in coordinated motion
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("swarm_1", 64, 64, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 10, 10, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=4), 54, 10, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 54, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=2), 54, 54, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 32, 10, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=1), 32, 54, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 54, 32, 40000, 1),
        ])),
        S("swarm_2", 64, 64, 150, 2, _setup(built=[
            (PREY_TEMPLATE, PreyBrain(), 28, 28, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 36, 28, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 28, 36, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 36, 36, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 32, 25, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 32, 39, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 25, 32, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 39, 32, 30000, 1),
        ], food=50)),
        S("swarm_3", 96, 64, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 25, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 39, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 15, 28, 50000, 1),
            (PREY_TEMPLATE, PreyBrain(), 70, 25, 30000, 2),
            (PREY_TEMPLATE, PreyBrain(), 70, 32, 30000, 2),
            (PREY_TEMPLATE, PreyBrain(), 70, 39, 30000, 2),
            (PREY_TEMPLATE, PreyBrain(), 75, 28, 30000, 2),
        ], food=30)),
        S("swarm_4", 96, 48, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 20, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 28, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 15, 24, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 20, 20, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 20, 28, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 25, 24, 60000, 1),
        ])),
        S("swarm_5", 64, 64, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 15, 40000, 1),
            (PREY_TEMPLATE, PreyBrain(), 50, 15, 30000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=5), 15, 50, 40000, 3),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 55, 50, 40000, 4),
            (PREY_TEMPLATE, PreyBrain(), 32, 32, 30000, 5),
        ], food=40)),
    ]

    # ────────────────────────────────────────────────────────────
    # 7. GARDEN — multiple plants growing together
    # ────────────────────────────────────────────────────────────
    def _garden_grid(engine: SimulationEngine) -> None:
        gid = 1
        for rr in range(3):
            for cc in range(3):
                q = 16 + cc * 16
                r = 16 + rr * 16
                engine.create_organism(q, r, PHOTO, 999999, gid, PlantBrain())
                gid = min(gid + 1, 5)

    def _garden_dense(engine: SimulationEngine) -> None:
        gid = 1
        for rr in range(4):
            for cc in range(4):
                q = 10 + cc * 20
                r = 10 + rr * 20
                engine.create_organism(q, r, PHOTO, 999999, gid, PlantBrain())
                gid = (gid % 5) + 1

    scenes += [
        S("garden_1", 64, 64, 150, 2, _garden_grid),
        S("garden_2", 96, 32, 150, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 10, 16, 999999, 1),
            (PHOTO, PlantBrain(), 28, 16, 999999, 2),
            (PHOTO, PlantBrain(), 46, 16, 999999, 3),
            (PHOTO, PlantBrain(), 64, 16, 999999, 4),
            (PHOTO, PlantBrain(), 82, 16, 999999, 5),
        ])),
        S("garden_3", 64, 64, 200, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 32, 32, 9999999, 1),
            (PHOTO, PlantBrain(), 12, 12, 200000, 2),
            (PHOTO, PlantBrain(), 52, 12, 200000, 3),
            (PHOTO, PlantBrain(), 12, 52, 200000, 4),
            (PHOTO, PlantBrain(), 52, 52, 200000, 5),
        ])),
        S("garden_4", 64, 64, 150, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 32, 10, 999999, 1),
            (PHOTO, PlantBrain(), 50, 22, 999999, 2),
            (PHOTO, PlantBrain(), 50, 42, 999999, 3),
            (PHOTO, PlantBrain(), 32, 54, 999999, 4),
            (PHOTO, PlantBrain(), 14, 42, 999999, 5),
            (PHOTO, PlantBrain(), 14, 22, 999999, 6),
        ])),
        S("garden_5", 96, 96, 200, 3, _garden_dense),
    ]

    # ────────────────────────────────────────────────────────────
    # 8. SIEGE — attacks on stationary targets
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("siege_1", 64, 64, 200, 2, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 32, 32, 99999, 1),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 54, 32, 50000, 2),
            ],
        )),
        S("siege_2", 64, 64, 200, 2, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 32, 32, 99999, 1),
                (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 10, 32, 50000, 2),
                (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 54, 32, 50000, 3),
            ],
        )),
        S("siege_3", 48, 48, 150, 2, _setup(built=[
            (PREY_TEMPLATE, PreyBrain(), 24, 24, 30000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 10, 10, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=4), 38, 10, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=1), 10, 38, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=2), 38, 38, 50000, 2),
        ])),
        S("siege_4", 48, 48, 200, 2, _setup(
            seeds=[(PHOTO, PlantBrain(), 24, 24, 9999999, 1)],
            built=[(PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 24, 50000, 2)],
        )),
        S("siege_5", 64, 64, 200, 2, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 32, 32, 99999, 1),
                (PREY_TEMPLATE, PreyBrain(), 30, 28, 30000, 1),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 30, 50000, 2),
            ],
            food=20,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 9. RACE — competition for resources
    # ────────────────────────────────────────────────────────────
    def _race_center_food(engine: SimulationEngine) -> None:
        prebuild_organism(engine, PREY_TEMPLATE, 10, 32, 30000, 1, PreyBrain())
        prebuild_organism(engine, PREY_TEMPLATE, 54, 32, 30000, 2, PreyBrain())
        _food_block(engine, 28, 28, 8, 8)

    def _race_line_food(engine: SimulationEngine) -> None:
        prebuild_organism(engine, PREY_TEMPLATE, 10, 24, 30000, 1, PreyBrain())
        prebuild_organism(engine, PREY_TEMPLATE, 86, 24, 30000, 2, PreyBrain())
        for q in range(20, 78, 2):
            place_food(engine, q, 24)

    scenes += [
        S("race_1", 64, 64, 200, 2, _race_center_food),
        S("race_2", 64, 64, 200, 2, _setup(
            built=[
                (PREY_TEMPLATE, PreyBrain(), 10, 10, 30000, 1),
                (PREY_TEMPLATE, PreyBrain(), 54, 10, 30000, 2),
                (PREY_TEMPLATE, PreyBrain(), 32, 54, 30000, 3),
            ],
            food=60,
        )),
        S("race_3", 96, 48, 200, 2, _race_line_food),
        S("race_4", 64, 64, 200, 2, _setup(
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 1),
                (PREY_TEMPLATE, PreyBrain(), 54, 32, 30000, 2),
            ],
            food=40,
        )),
        S("race_5", 64, 64, 200, 2, _setup(
            built=[
                (PREY_TEMPLATE, PreyBrain(), 10, 32, 20000, 1),
                (PREY_TEMPLATE, PreyBrain(), 32, 10, 20000, 2),
                (PREY_TEMPLATE, PreyBrain(), 54, 32, 20000, 3),
                (PREY_TEMPLATE, PreyBrain(), 32, 54, 20000, 4),
            ],
            food=50,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 10. MIGRATION — organisms traversing the grid
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("migrate_1", 96, 48, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 18, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 24, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 30, 60000, 1),
        ])),
        S("migrate_2", 64, 64, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 28, 60000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 36, 60000, 1),
            (PREY_TEMPLATE, PreyBrain(), 54, 28, 40000, 2),
            (PREY_TEMPLATE, PreyBrain(), 54, 36, 40000, 2),
        ], food=30)),
        S("migrate_3", 96, 64, 250, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 80000, 1),
            (PREY_TEMPLATE, PreyBrain(), 50, 32, 50000, 2),
            (PREY_TEMPLATE, PreyBrain(), 55, 28, 50000, 2),
        ], food=20)),
        S("migrate_4", 128, 32, 300, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 16, 100000, 1),
        ])),
        S("migrate_5", 64, 64, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 32, 32, 40000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 32, 32, 40000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=1), 32, 32, 40000, 3),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=4), 32, 32, 40000, 4),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 11. COLONY — reproduction creating settlements
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("colony_1", 64, 64, 300, 3, _setup(
            seeds=[(PHOTO, ReproducerBrain(growth_ticks=30, reproduce_threshold=400), 32, 32, 20000, 1)],
            food=80,
        )),
        S("colony_2", 96, 96, 400, 4, _setup(
            seeds=[(PHOTO, ReproducerBrain(growth_ticks=25, reproduce_threshold=250), 48, 48, 100000, 1)],
            food=150,
        )),
        S("colony_3", 96, 64, 350, 3, _setup(
            seeds=[
                (PHOTO, ReproducerBrain(growth_ticks=30, reproduce_threshold=400), 25, 32, 15000, 1),
                (PHOTO, ReproducerBrain(growth_ticks=30, reproduce_threshold=400), 71, 32, 15000, 2),
            ],
            food=100,
        )),
        S("colony_4", 32, 32, 250, 2, _setup(
            seeds=[(PHOTO, ReproducerBrain(growth_ticks=20, reproduce_threshold=300), 16, 16, 10000, 1)],
            food=40,
        )),
        S("colony_5", 64, 64, 400, 4, _setup(
            seeds=[(PHOTO, ReproducerBrain(growth_ticks=40, reproduce_threshold=500), 32, 32, 8000, 1)],
            food=60,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 12. STANDOFF — organisms facing each other at distance
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("standoff_1", 48, 48, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 12, 24, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 36, 24, 50000, 2),
        ])),
        S("standoff_2", 48, 48, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 12, 24, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 36, 24, 50000, 2),
        ])),
        S("standoff_3", 48, 48, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 12, 24, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=3), 36, 24, 50000, 2),
        ])),
        S("standoff_4", 64, 64, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=5), 32, 10, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=1), 10, 50, 50000, 2),
            (PREY_TEMPLATE, PreyBrain(), 54, 50, 30000, 3),
        ], food=20)),
        S("standoff_5", 64, 64, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=5), 20, 10, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=4), 44, 10, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=1), 10, 50, 50000, 3),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=2), 54, 50, 50000, 4),
            (PREY_TEMPLATE, PreyBrain(), 32, 32, 30000, 5),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 13. FEAST — organisms eating abundant resources
    # ────────────────────────────────────────────────────────────
    def _feast_field(engine: SimulationEngine) -> None:
        prebuild_organism(engine, PREDATOR_TEMPLATE, 32, 32, 50000, 1,
                          PredatorBrain(seek_direction=0))
        _food_block(engine, 10, 10, 44, 44)

    def _feast_prey(engine: SimulationEngine) -> None:
        prebuild_organism(engine, PREY_TEMPLATE, 32, 32, 30000, 1, PreyBrain())
        _food_block(engine, 5, 5, 54, 54)

    def _feast_group(engine: SimulationEngine) -> None:
        for pq, pr, gid in [(20, 20, 1), (44, 20, 2), (20, 44, 3), (44, 44, 4)]:
            prebuild_organism(engine, PREY_TEMPLATE, pq, pr, 20000, gid, PreyBrain())
        _food_block(engine, 10, 10, 44, 44)

    scenes += [
        S("feast_1", 64, 64, 200, 2, _feast_field),
        S("feast_2", 64, 64, 200, 2, _feast_prey),
        S("feast_3", 64, 64, 200, 2, _setup(
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 1),
                (PREY_TEMPLATE, PreyBrain(), 40, 32, 30000, 2),
            ],
            food=80,
        )),
        S("feast_4", 64, 64, 200, 2, _feast_group),
        S("feast_5", 48, 48, 200, 2, _setup(
            seeds=[(PHOTO, PlantBrain(), 24, 24, 999999, 1)],
            food=100,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 14. INVASION — species entering another's territory
    # ────────────────────────────────────────────────────────────
    def _invade_garden(engine: SimulationEngine) -> None:
        for r in range(3):
            for c in range(3):
                engine.create_organism(
                    40 + c * 10, 15 + r * 10, PHOTO, 999999, 1, PlantBrain(),
                )
        prebuild_organism(engine, PREDATOR_TEMPLATE, 10, 25, 50000, 2,
                          PredatorBrain(seek_direction=0))
        prebuild_organism(engine, PREDATOR_TEMPLATE, 10, 35, 50000, 2,
                          PredatorBrain(seek_direction=0))

    def _invade_two_sides(engine: SimulationEngine) -> None:
        for rr in range(3):
            engine.create_organism(24 + rr * 8, 24 + rr * 8, PHOTO, 999999, 1, PlantBrain())
        prebuild_organism(engine, WARRIOR_TEMPLATE, 10, 32, 50000, 2,
                          WarriorBrain(seek_direction=0))
        prebuild_organism(engine, WARRIOR_TEMPLATE, 54, 32, 50000, 3,
                          WarriorBrain(seek_direction=3))

    scenes += [
        S("invade_1", 96, 48, 250, 2, _invade_garden),
        S("invade_2", 64, 64, 200, 2, _invade_two_sides),
        S("invade_3", 64, 64, 250, 3, _setup(
            seeds=[(PHOTO, ReproducerBrain(growth_ticks=30, reproduce_threshold=400), 20, 32, 10000, 1)],
            built=[(PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 55, 32, 50000, 2)],
            food=40,
        )),
        S("invade_4", 96, 64, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 28, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 36, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 15, 32, 50000, 1),
            (PLANT_TEMPLATE, PlantBrain(), 60, 32, 99999, 2),
            (PREY_TEMPLATE, PreyBrain(), 70, 28, 30000, 3),
            (PREY_TEMPLATE, PreyBrain(), 70, 36, 30000, 3),
        ], food=30)),
        S("invade_5", 96, 96, 300, 3, _setup(
            built=[
                (PLANT_TEMPLATE, PlantBrain(), 25, 48, 99999, 1),
                (PREY_TEMPLATE, PreyBrain(), 30, 40, 30000, 1),
                (PREY_TEMPLATE, PreyBrain(), 20, 55, 30000, 1),
                (PLANT_TEMPLATE, PlantBrain(), 70, 48, 99999, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 75, 40, 50000, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 65, 55, 50000, 2),
            ],
            food=50,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 15. SYMMETRY — body plan showcase
    # ────────────────────────────────────────────────────────────
    def _sym_all(engine: SimulationEngine) -> None:
        configs = [
            (PLANT_TEMPLATE, PlantBrain(), 12, 24, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(), 30, 24, 2),
            (PREY_TEMPLATE, PreyBrain(), 48, 24, 3),
            (WARRIOR_TEMPLATE, WarriorBrain(), 66, 24, 4),
            (REPRODUCER_TEMPLATE, ReproducerBrain(), 84, 24, 5),
        ]
        for tmpl, brain, q, r, gid in configs:
            prebuild_organism(engine, tmpl, q, r, 999999, gid, brain)

    scenes += [
        S("sym_1", 96, 48, 100, 1, _sym_all),
        S("sym_2", 24, 24, 80, 1, _setup(built=[
            (PLANT_TEMPLATE, PlantBrain(), 12, 12, 999999, 1),
        ])),
        S("sym_3", 24, 24, 80, 1, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(), 12, 12, 999999, 1),
        ])),
        S("sym_4", 24, 24, 80, 1, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(), 12, 12, 999999, 1),
        ])),
        S("sym_5", 64, 32, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 16, 60000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 40, 16, 60000, 2),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 16. SPEED — fast vs slow demonstrations
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("speed_1", 96, 48, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 24, 60000, 1),
            (PREY_TEMPLATE, PreyBrain(), 80, 24, 40000, 2),
        ])),
        S("speed_2", 96, 48, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 12, 60000, 1),
            (PREY_TEMPLATE, PreyBrain(), 10, 24, 40000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 10, 36, 60000, 3),
        ])),
        S("speed_3", 128, 32, 250, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 16, 100000, 1),
        ])),
        S("speed_4", 64, 64, 200, 2, _setup(
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 1),
            ],
            seeds=[
                (PHOTO, PlantBrain(), 50, 32, 999999, 2),
            ],
        )),
        S("speed_5", 64, 48, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 24, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 54, 24, 50000, 2),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 17. DEFENSE — armored organisms resisting attack
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("defense_1", 48, 48, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(), 24, 24, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 24, 50000, 2),
        ])),
        S("defense_2", 64, 64, 200, 2, _setup(
            built=[
                (WARRIOR_TEMPLATE, WarriorBrain(), 32, 32, 50000, 1),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 28, 50000, 2),
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 10, 36, 50000, 2),
            ],
        )),
        S("defense_3", 96, 48, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(), 30, 24, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(), 40, 24, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(), 50, 24, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 24, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 80, 24, 50000, 3),
        ])),
        S("defense_4", 48, 48, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(), 20, 24, 50000, 1),
            (PREY_TEMPLATE, PreyBrain(), 14, 24, 30000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 38, 24, 50000, 2),
        ])),
        S("defense_5", 64, 64, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(), 32, 32, 80000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=5), 15, 15, 40000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=4), 49, 15, 40000, 3),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=1), 15, 49, 40000, 4),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=2), 49, 49, 40000, 5),
        ])),
    ]

    # ────────────────────────────────────────────────────────────
    # 18. EXTINCTION — species being eliminated
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("extinct_1", 64, 64, 250, 2, _setup(
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 80000, 1),
                (PREY_TEMPLATE, PreyBrain(), 40, 25, 20000, 2),
                (PREY_TEMPLATE, PreyBrain(), 45, 32, 20000, 2),
                (PREY_TEMPLATE, PreyBrain(), 40, 39, 20000, 2),
            ],
            food=20,
        )),
        S("extinct_2", 48, 48, 200, 2, _setup(built=[
            (PREY_TEMPLATE, PreyBrain(), 16, 16, 5000, 1),
            (PREY_TEMPLATE, PreyBrain(), 32, 16, 5000, 2),
            (PREY_TEMPLATE, PreyBrain(), 16, 32, 5000, 3),
            (PREY_TEMPLATE, PreyBrain(), 32, 32, 5000, 4),
        ])),
        S("extinct_3", 48, 48, 200, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=5), 16, 8, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=4), 32, 8, 50000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=1), 16, 40, 50000, 3),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=2), 32, 40, 50000, 4),
        ])),
        S("extinct_4", 48, 48, 200, 2, _setup(built=[
            (PREY_TEMPLATE, PreyBrain(), 24, 24, 30000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 20, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 40, 20, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 28, 50000, 2),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=3), 40, 28, 50000, 2),
        ])),
        S("extinct_5", 64, 64, 300, 3, _setup(
            seeds=[(PHOTO, PlantBrain(), 32, 32, 999999, 1)],
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 30000, 2),
            ],
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 19. CROWD — dense population dynamics
    # ────────────────────────────────────────────────────────────
    def _crowd_arena(engine: SimulationEngine) -> None:
        positions = [
            (8, 8), (16, 8), (24, 8),
            (8, 16), (16, 16), (24, 16),
            (8, 24), (16, 24), (24, 24),
        ]
        brains = [PreyBrain(), PredatorBrain(seek_direction=0), WarriorBrain()]
        for i, (q, r) in enumerate(positions):
            tmpl = [PREY_TEMPLATE, PREDATOR_TEMPLATE, WARRIOR_TEMPLATE][i % 3]
            brain = brains[i % 3]
            gid = (i % 3) + 1
            prebuild_organism(engine, tmpl, q, r, 30000, gid, brain)
        scatter_food(engine, 30)

    def _crowd_battle(engine: SimulationEngine) -> None:
        positions = [
            (8, 8), (24, 8), (8, 24), (24, 24),
            (16, 16), (8, 16), (24, 16), (16, 8), (16, 24),
        ]
        for i, (q, r) in enumerate(positions):
            gid = (i % 5) + 1
            prebuild_organism(engine, WARRIOR_TEMPLATE, q, r, 40000, gid,
                              WarriorBrain())

    scenes += [
        S("crowd_1", 32, 32, 150, 2, _crowd_arena),
        S("crowd_2", 32, 32, 150, 2, _crowd_battle),
        S("crowd_3", 32, 32, 150, 2, _setup(seeds=[
            (PHOTO, PlantBrain(), 8, 8, 999999, 1),
            (PHOTO, PlantBrain(), 24, 8, 999999, 2),
            (PHOTO, PlantBrain(), 8, 24, 999999, 3),
            (PHOTO, PlantBrain(), 24, 24, 999999, 4),
            (PHOTO, PlantBrain(), 16, 16, 999999, 5),
        ])),
        S("crowd_4", 32, 32, 150, 2, _setup(
            built=[
                (PREDATOR_TEMPLATE, PredatorBrain(), 8, 8, 30000, 1),
                (PREY_TEMPLATE, PreyBrain(), 24, 8, 20000, 2),
                (WARRIOR_TEMPLATE, WarriorBrain(), 8, 24, 30000, 3),
                (PREDATOR_TEMPLATE, PredatorBrain(), 24, 24, 30000, 4),
            ],
            seeds=[(PHOTO, PlantBrain(), 16, 16, 999999, 5)],
            food=20,
        )),
        S("crowd_5", 32, 32, 250, 2, _setup(
            seeds=[(PHOTO, ReproducerBrain(growth_ticks=20, reproduce_threshold=200), 16, 16, 20000, 1)],
            food=60,
        )),
    ]

    # ────────────────────────────────────────────────────────────
    # 20. FORMATION — organisms in geometric patterns
    # ────────────────────────────────────────────────────────────
    scenes += [
        S("form_1", 96, 32, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 10, 16, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 22, 16, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 34, 16, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 46, 16, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(seek_direction=0), 58, 16, 50000, 1),
        ])),
        S("form_2", 64, 64, 150, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 20, 32, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 14, 26, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 14, 38, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 20, 50000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 8, 44, 50000, 1),
        ])),
        S("form_3", 64, 64, 150, 2, _setup(built=[
            (WARRIOR_TEMPLATE, WarriorBrain(), 32, 16, 50000, 1),
            (WARRIOR_TEMPLATE, WarriorBrain(), 44, 24, 50000, 2),
            (WARRIOR_TEMPLATE, WarriorBrain(), 44, 40, 50000, 3),
            (WARRIOR_TEMPLATE, WarriorBrain(), 32, 48, 50000, 4),
            (WARRIOR_TEMPLATE, WarriorBrain(), 20, 40, 50000, 5),
            (WARRIOR_TEMPLATE, WarriorBrain(), 20, 24, 50000, 6),
        ])),
        S("form_4", 64, 48, 200, 2, _setup(built=[
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 16, 50000, 1),
            (PREY_TEMPLATE, PreyBrain(), 18, 16, 30000, 1),
            (PREDATOR_TEMPLATE, PredatorBrain(seek_direction=0), 10, 32, 50000, 2),
            (PREY_TEMPLATE, PreyBrain(), 18, 32, 30000, 2),
        ], food=30)),
        S("form_5", 64, 64, 150, 2, _setup(built=[
            (PREY_TEMPLATE, PreyBrain(), 32, 12, 30000, 1),
            (PREY_TEMPLATE, PreyBrain(), 48, 22, 30000, 2),
            (PREY_TEMPLATE, PreyBrain(), 48, 42, 30000, 3),
            (PREY_TEMPLATE, PreyBrain(), 32, 52, 30000, 4),
            (PREY_TEMPLATE, PreyBrain(), 16, 42, 30000, 5),
            (PREY_TEMPLATE, PreyBrain(), 16, 22, 30000, 6),
        ], food=40)),
    ]

    return scenes


# ── Runner ───────────────────────────────────────────────────────────

SCENE_LIST = all_scenes()
SCENE_MAP: dict[str, SceneConfig] = {s.name: s for s in SCENE_LIST}

# Group by concept prefix (e.g. "growth" matches "growth_1" through "growth_5")
CONCEPTS: dict[str, list[SceneConfig]] = {}
for s in SCENE_LIST:
    prefix = s.name.rsplit("_", 1)[0]
    CONCEPTS.setdefault(prefix, []).append(s)


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
            orgs = frames[-1].get("organisms", [])
            alive = sum(1 for f in orgs if f.get("alive"))
            print(f"  tick {tick}/{scene.tick_limit} ({elapsed:.1f}s) — {alive} organisms alive")

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
    import argparse
    parser = argparse.ArgumentParser(description="Run showcase demo scenes")
    parser.add_argument("filter", nargs="?", help="Scene name, concept prefix, or 'all'")
    parser.add_argument("--render", action="store_true", help="Also render PNGs and compile MP4s")
    parser.add_argument("--list", action="store_true", help="List all scenes")
    args = parser.parse_args()

    if args.list:
        for concept, concept_scenes in CONCEPTS.items():
            print(f"\n{concept}:")
            for s in concept_scenes:
                print(f"  {s.name} ({s.width}x{s.height}, {s.tick_limit} ticks)")
        print(f"\nTotal: {len(SCENE_LIST)} scenes across {len(CONCEPTS)} concepts")
        return

    # Determine which scenes to run
    if args.filter:
        if args.filter in SCENE_MAP:
            scenes = [SCENE_MAP[args.filter]]
        elif args.filter in CONCEPTS:
            scenes = CONCEPTS[args.filter]
        else:
            print(f"Unknown scene/concept: {args.filter}")
            print(f"Concepts: {', '.join(CONCEPTS.keys())}")
            sys.exit(1)
    else:
        scenes = SCENE_LIST

    print(f"Running {len(scenes)} showcase scene(s)...")
    t_total = time.perf_counter()
    paths: list[Path] = []
    for i, scene in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}]", end="")
        paths.append(run_scene(scene))

    elapsed_total = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"All done! {len(paths)} scenes in {elapsed_total:.1f}s")
    print(f"Output: {OUTPUT_DIR}/")

    if args.render:
        print(f"\n{'='*60}")
        print("Rendering frames and compiling videos...")
        _render_and_compile(paths)


def _render_and_compile(paths: list[Path]) -> None:
    """Render PNGs and compile MP4s for all generated scenes."""
    from demos.render_frames import render_scene as render_scene_pngs
    import subprocess

    for json_path in paths:
        name = json_path.stem
        render_scene_pngs(name)

        scene_dir = OUTPUT_DIR / name
        mp4_path = OUTPUT_DIR / f"{name}.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "15",
            "-i", str(scene_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(mp4_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            size_kb = mp4_path.stat().st_size / 1024
            print(f"  {name}.mp4 ({size_kb:.0f} KB)")
        else:
            print(f"  ffmpeg failed for {name}: {result.stderr[:200]}")


if __name__ == "__main__":
    main()

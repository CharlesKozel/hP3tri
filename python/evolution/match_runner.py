from __future__ import annotations

import time as _time
from typing import Any

import numpy as np

from simulator.cell_types import CellType, CELL_PROPERTIES
from simulator.engine import SimulationEngine
from body_plans.cppn import CppnBodyPlan, CPPN_TOTAL_WEIGHTS, CPPN_NUM_HIDDEN
from brains.rule_brain import RuleBrain


_cached_engine: SimulationEngine | None = None


def _get_engine(width: int, height: int, seed: int) -> SimulationEngine:
    """Get or create a reusable engine (avoids Taichi recompilation per match)."""
    global _cached_engine
    if (_cached_engine is not None
            and _cached_engine.width == width
            and _cached_engine.height == height):
        _cached_engine.reset(seed)
        return _cached_engine
    _cached_engine = SimulationEngine(width, height, seed)
    return _cached_engine


def _setup_genome(
    engine: SimulationEngine,
    genome_dict: dict[str, Any],
) -> tuple[int, CppnBodyPlan, RuleBrain, int]:
    """Register a genome's body plan, brain, and template with the engine.

    Returns (genome_id, body_plan, brain, seed_cell_type).
    """
    gid = int(genome_dict["id"])
    weights = np.array(genome_dict["cppn_weights"], dtype=np.float32)
    activations = np.array(genome_dict["cppn_activations"], dtype=np.int32)
    symmetry = int(genome_dict.get("symmetry_mode", 1))
    brain_params = np.array(genome_dict["brain_params"], dtype=np.float32)

    body_plan = CppnBodyPlan(weights, activations, symmetry)
    brain = RuleBrain(brain_params)

    engine.set_genome_brain_params(gid, brain_params)

    template = body_plan.generate_template(dev_time=0.0)
    genome_data: dict[str, Any] = {"body_template": template}
    engine.genome_registry[gid] = genome_data

    seed_cell_type = int(genome_dict.get("seed_cell_type", int(CellType.SOFT_TISSUE)))

    return gid, body_plan, brain, seed_cell_type


def _find_empty_cell(
    engine: SimulationEngine,
    rng: np.random.Generator,
    margin: int = 4,
    max_attempts: int = 200,
) -> tuple[int, int] | None:
    """Find a random empty tile within the grid (with margin from edges)."""
    ct_np = engine.grid.cell_type.to_numpy()
    oid_np = engine.grid.organism_id.to_numpy()
    for _ in range(max_attempts):
        q = int(rng.integers(margin, engine.width - margin))
        r = int(rng.integers(margin, engine.height - margin))
        idx = r * engine.width + q
        if int(ct_np[idx]) == 0 and int(oid_np[idx]) == 0:
            return q, r
    return None


def _place_seeds(
    engine: SimulationEngine,
    genome_infos: list[tuple[int, CppnBodyPlan, RuleBrain, int]],
    population_size: int,
    rng: np.random.Generator,
) -> dict[int, list[int]] | str:
    """Place population_size seed organisms per genome, alternating 1-by-1.

    Returns org_ids_by_genome on success, or an error string on failure.
    """
    total_needed = population_size * len(genome_infos)
    usable_area = (engine.width - 8) * (engine.height - 8)
    if total_needed > usable_area:
        return (
            f"Cannot place {total_needed} organisms "
            f"in {usable_area} usable tiles"
        )

    org_ids_by_genome: dict[int, list[int]] = {}
    for gid, _, _, _ in genome_infos:
        org_ids_by_genome[gid] = []

    for _ in range(population_size):
        for gid, body_plan, brain, seed_cell_type in genome_infos:
            pos = _find_empty_cell(engine, rng)
            if pos is None:
                return (
                    f"Failed to place organism for genome {gid}: "
                    f"grid too crowded after {len(org_ids_by_genome[gid])} placements"
                )
            q, r = pos

            org_id = engine.create_organism(
                seed_q=q, seed_r=r,
                seed_cell_type=seed_cell_type,
                starting_energy=800,
                genome_id=gid,
                brain=brain,
                body_plan=body_plan,
            )

            genome_data = engine.genome_registry[gid]
            genome_data["origin_q"] = q
            genome_data["origin_r"] = r

            org_ids_by_genome[gid].append(org_id)

    return org_ids_by_genome


def _run_match(
    config: dict[str, Any],
    genome_dicts: list[dict[str, Any]],
    snapshot_interval: int = 0,
) -> tuple[SimulationEngine, list[int], list[dict] | None, str | None]:
    """Core match runner shared by evolution and visualization paths.

    Returns (engine, genome_ids, replay_frames_or_None, error_or_None).
    snapshot_interval=0 disables replay recording.
    """
    width: int = config.get("width", 64)
    height: int = config.get("height", 64)
    tick_limit: int = config.get("tick_limit", 500)
    seed: int = config.get("seed", 42)
    food_count: int = config.get("food_count", 80)
    food_respawn_rate: int = config.get("food_respawn_rate", 5)
    food_respawn_interval: int = config.get("food_respawn_interval", 20)
    population_size: int = config.get("population_size", 1)

    rng = np.random.default_rng(seed)
    engine = _get_engine(width, height, seed)

    _t0 = _time.perf_counter()
    genome_infos: list[tuple[int, CppnBodyPlan, RuleBrain, int]] = []
    genome_ids: list[int] = []
    for genome_dict in genome_dicts:
        gid, body_plan, brain, seed_ct = _setup_genome(engine, genome_dict)
        genome_infos.append((gid, body_plan, brain, seed_ct))
        genome_ids.append(gid)
    print(f"  [Match] {len(genome_dicts)} genomes setup ({_time.perf_counter() - _t0:.2f}s)", flush=True)

    result = _place_seeds(engine, genome_infos, population_size, rng)
    if isinstance(result, str):
        return engine, genome_ids, None, result

    _place_food(engine, food_count, rng)
    print(f"  [Match] Seeds + food placed ({_time.perf_counter() - _t0:.2f}s)", flush=True)

    replay: list[dict] | None = None
    if snapshot_interval > 0:
        replay = []
        engine.recompute_aggregates()
        replay.append(engine.snapshot())
    else:
        engine.recompute_aggregates()

    _tick_t0 = _time.perf_counter()
    for tick in range(tick_limit):
        engine.step()

        if replay is not None and (tick % snapshot_interval == 0 or tick == tick_limit - 1):
            replay.append(engine.snapshot())

        if food_respawn_interval > 0 and tick > 0 and tick % food_respawn_interval == 0:
            _place_food(engine, food_respawn_rate, rng)

    elapsed = _time.perf_counter() - _tick_t0
    tps = tick_limit / elapsed if elapsed > 0 else 0
    print(f"  [Match] {tick_limit} ticks in {elapsed:.2f}s ({tps:.0f} ticks/s)", flush=True)
    return engine, genome_ids, replay, None


def run_evolution_match(
    config: dict[str, Any],
    genome_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    engine, genome_ids, _, error = _run_match(config, genome_dicts)
    if error is not None:
        return [{"error": error}]

    results: list[dict[str, Any]] = []
    for gid in genome_ids:
        final_cells = _count_genome_cells(engine, gid)
        final_energy = _sum_genome_energy(engine, gid)
        mobility, aggression = _compute_behavior_descriptors(engine, gid)

        results.append({
            "genome_id": gid,
            "final_cell_count": final_cells,
            "survived": final_cells > 0,
            "peak_cell_count": final_cells,
            "final_energy": final_energy,
            "mobility": float(mobility),
            "aggression": float(aggression),
        })

    return results


def run_visualizable_match(
    config: dict[str, Any],
    genome_dicts: list[dict[str, Any]],
) -> list[dict]:
    snapshot_interval: int = config.get("snapshot_interval", 5)
    _, _, replay, _ = _run_match(config, genome_dicts, snapshot_interval)
    return replay if replay is not None else []


def _place_food(engine: SimulationEngine, count: int, rng: np.random.Generator) -> None:
    ct_np = engine.grid.cell_type.to_numpy()
    oid_np = engine.grid.organism_id.to_numpy()
    placed = 0
    attempts = 0
    max_attempts = count * 10
    while placed < count and attempts < max_attempts:
        q = int(rng.integers(0, engine.width))
        r = int(rng.integers(0, engine.height))
        idx = r * engine.width + q
        if int(ct_np[idx]) == 0 and int(oid_np[idx]) == 0:
            engine.grid[idx].cell_type = int(CellType.FOOD)
            ct_np[idx] = int(CellType.FOOD)
            placed += 1
        attempts += 1


def _count_genome_cells(engine: SimulationEngine, genome_id: int) -> int:
    total = 0
    for oid in range(1, engine.next_org_id):
        if (engine.organism_genome_map.get(oid) == genome_id
                and engine.organisms[oid].alive == 1):
            total += int(engine.organisms[oid].cell_count)
    return total


def _sum_genome_energy(engine: SimulationEngine, genome_id: int) -> int:
    total = 0
    for oid in range(1, engine.next_org_id):
        if (engine.organism_genome_map.get(oid) == genome_id
                and engine.organisms[oid].alive == 1):
            total += int(engine.organisms[oid].energy)
    return total


def _compute_behavior_descriptors(
    engine: SimulationEngine,
    genome_id: int,
) -> tuple[float, float]:
    total_locomotion = 0
    total_mass = 0
    total_mouth_spike = 0
    total_cells = 0
    count = 0
    for oid in range(1, engine.next_org_id):
        if engine.organism_genome_map.get(oid) == genome_id and engine.organisms[oid].alive == 1:
            total_locomotion += int(engine.organisms[oid].locomotion_power)
            total_mass += int(engine.organisms[oid].total_mass)
            cc = int(engine.organisms[oid].cell_count)
            total_cells += cc
            mouth_ct = int(engine.organisms[oid].cell_type_counts[int(CellType.MOUTH)])
            spike_ct = int(engine.organisms[oid].cell_type_counts[int(CellType.SPIKE)])
            total_mouth_spike += mouth_ct + spike_ct
            count += 1

    if count == 0 or total_mass == 0 or total_cells == 0:
        return 0.0, 0.0

    mobility = min(total_locomotion / total_mass, 1.0)
    aggression = min(total_mouth_spike / total_cells, 1.0)
    return mobility, aggression


def run_genome_preview(
    config: dict[str, Any],
    genome_dict: dict[str, Any],
) -> dict[str, Any]:
    """Generate a preview by evaluating the CPPN template directly (no simulation)."""
    width: int = config.get("width", 64)
    height: int = config.get("height", 64)

    weights = np.array(genome_dict["cppn_weights"], dtype=np.float32)
    activations = np.array(genome_dict["cppn_activations"], dtype=np.int32)
    symmetry = int(genome_dict.get("symmetry_mode", 1))

    body_plan = CppnBodyPlan(weights, activations, symmetry)
    template = body_plan.generate_template(dev_time=0.0)

    center_q = width // 2
    center_r = height // 2

    tiles: list[dict[str, Any]] = []
    for (dq, dr), cell_type in template.items():
        tiles.append({
            "q": center_q + dq,
            "r": center_r + dr,
            "terrainType": 0,
            "cellType": int(cell_type),
            "organismId": 1,
        })

    snapshot: dict[str, Any] = {
        "tick": 0,
        "status": "preview",
        "grid": {
            "width": width,
            "height": height,
            "tiles": tiles,
        },
        "organisms": [{
            "id": 1,
            "genomeId": int(genome_dict["id"]),
            "energy": 999999,
            "alive": True,
            "cellCount": len(tiles),
        }],
    }

    return {
        "final_snapshot": snapshot,
        "snapshots": [snapshot],
        "final_cell_count": len(tiles),
        "survived": len(tiles) > 0,
    }


def run_genome_previews_batch(
    config: dict[str, Any],
    genome_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [run_genome_preview(config, gd) for gd in genome_dicts]

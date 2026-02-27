from __future__ import annotations

from typing import Any

import numpy as np

from simulator.cell_types import CellType, CELL_PROPERTIES
from simulator.engine import SimulationEngine
from body_plans.cppn import CppnBodyPlan, CPPN_TOTAL_WEIGHTS, CPPN_NUM_HIDDEN
from brains.rule_brain import RuleBrain


def run_evolution_match(
    config: dict[str, Any],
    genome_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run a single evolution match and return per-genome results.

    Called from Kotlin via Jep.
    """
    width: int = config.get("width", 64)
    height: int = config.get("height", 64)
    tick_limit: int = config.get("tick_limit", 500)
    seed: int = config.get("seed", 42)
    food_count: int = config.get("food_count", 80)
    food_respawn_rate: int = config.get("food_respawn_rate", 5)
    food_respawn_interval: int = config.get("food_respawn_interval", 20)

    rng = np.random.default_rng(seed)
    engine = SimulationEngine(width, height, seed)

    genome_id_to_genome: dict[int, dict[str, Any]] = {}
    org_ids_by_genome: dict[int, list[int]] = {}

    for genome_dict in genome_dicts:
        gid = int(genome_dict["id"])
        genome_id_to_genome[gid] = genome_dict

        weights = np.array(genome_dict["cppn_weights"], dtype=np.float32)
        activations = np.array(genome_dict["cppn_activations"], dtype=np.int32)
        symmetry = int(genome_dict.get("symmetry_mode", 1))
        brain_params = np.array(genome_dict["brain_params"], dtype=np.float32)

        body_plan = CppnBodyPlan(weights, activations, symmetry)
        brain = RuleBrain(brain_params)

        template = body_plan.generate_template(dev_time=0.0)

        genome_data: dict[str, Any] = {
            "body_template": template,
        }
        engine.genome_registry[gid] = genome_data

        q = int(rng.integers(4, width - 4))
        r = int(rng.integers(4, height - 4))
        seed_ct = int(CellType.SOFT_TISSUE)

        org_id = engine.create_organism(
            seed_q=q, seed_r=r,
            seed_cell_type=seed_ct,
            starting_energy=200,
            genome_id=gid,
            brain=brain,
            body_plan=body_plan,
        )

        genome_data["origin_q"] = q
        genome_data["origin_r"] = r

        org_ids_by_genome.setdefault(gid, []).append(org_id)

    _place_food(engine, food_count, rng)

    peak_cells: dict[int, int] = {gid: 0 for gid in genome_id_to_genome}

    engine.recompute_aggregates()
    for tick in range(tick_limit):
        engine.step()

        for gid in genome_id_to_genome:
            total = _count_genome_cells(engine, gid)
            if total > peak_cells[gid]:
                peak_cells[gid] = total

        if food_respawn_interval > 0 and tick > 0 and tick % food_respawn_interval == 0:
            _place_food(engine, food_respawn_rate, rng)

    results: list[dict[str, Any]] = []
    for gid, genome_dict in genome_id_to_genome.items():
        final_cells = _count_genome_cells(engine, gid)
        final_energy = _sum_genome_energy(engine, gid)
        survived = final_cells > 0
        mobility, aggression = _compute_behavior_descriptors(engine, gid)

        results.append({
            "genome_id": gid,
            "final_cell_count": final_cells,
            "survived": survived,
            "peak_cell_count": peak_cells[gid],
            "final_energy": final_energy,
            "mobility": float(mobility),
            "aggression": float(aggression),
        })

    return results


def run_visualizable_match(
    config: dict[str, Any],
    genome_dicts: list[dict[str, Any]],
) -> list[dict]:
    """Run a match and return full replay frames for visualization."""
    width: int = config.get("width", 64)
    height: int = config.get("height", 64)
    tick_limit: int = config.get("tick_limit", 200)
    seed: int = config.get("seed", 42)
    food_count: int = config.get("food_count", 40)
    food_respawn_rate: int = config.get("food_respawn_rate", 3)
    food_respawn_interval: int = config.get("food_respawn_interval", 20)

    rng = np.random.default_rng(seed)
    engine = SimulationEngine(width, height, seed)

    for genome_dict in genome_dicts:
        gid = int(genome_dict["id"])

        weights = np.array(genome_dict["cppn_weights"], dtype=np.float32)
        activations = np.array(genome_dict["cppn_activations"], dtype=np.int32)
        symmetry = int(genome_dict.get("symmetry_mode", 1))
        brain_params = np.array(genome_dict["brain_params"], dtype=np.float32)

        body_plan = CppnBodyPlan(weights, activations, symmetry)
        brain = RuleBrain(brain_params)

        template = body_plan.generate_template(dev_time=0.0)
        genome_data: dict[str, Any] = {"body_template": template}
        engine.genome_registry[gid] = genome_data

        q = int(rng.integers(4, width - 4))
        r = int(rng.integers(4, height - 4))
        seed_ct = int(CellType.SOFT_TISSUE)

        engine.create_organism(
            seed_q=q, seed_r=r,
            seed_cell_type=seed_ct,
            starting_energy=200,
            genome_id=gid,
            brain=brain,
            body_plan=body_plan,
        )

        genome_data["origin_q"] = q
        genome_data["origin_r"] = r

    _place_food(engine, food_count, rng)

    replay: list[dict] = []
    engine.recompute_aggregates()
    replay.append(engine.snapshot())

    for tick in range(tick_limit):
        engine.step()
        replay.append(engine.snapshot())

        if food_respawn_interval > 0 and tick > 0 and tick % food_respawn_interval == 0:
            _place_food(engine, food_respawn_rate, rng)

    return replay


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

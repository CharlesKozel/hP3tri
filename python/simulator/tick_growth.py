from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import numpy as np
import taichi as ti

from simulator.hex_grid import TerrainType, NEIGHBOR_OFFSETS, neighbors, wrap
from simulator.cell_types import CELL_PROPERTIES, CellType
from simulator.types import OrganismId
from interfaces.brain import OrganismView

if TYPE_CHECKING:
    from simulator.engine import SimulationEngine

MAX_ORGANISM_CELLS = 1000
MID_PRIORITY_SCALE: int = 1_000_000

BASE_REPRODUCTION_COST = 20
DEFAULT_OFFSPRING_ENERGY = 30


class _GrowthClaim(NamedTuple):
    org_id: OrganismId
    q: int
    r: int
    cell_type: int
    priority: float


class _ReproClaim(NamedTuple):
    parent_id: OrganismId
    q: int
    r: int
    seed_cell_type: int
    total_cost: int


def execute_growth(engine: SimulationEngine) -> None:
    grid_ct = engine.grid.cell_type.to_numpy()
    grid_oid = engine.grid.organism_id.to_numpy()
    grid_tt = engine.grid.terrain_type.to_numpy()

    growth_requests: list[_GrowthClaim] = []
    reproduction_requests: list[_ReproClaim] = []

    for org_idx in range(engine.organism_manager.highest_id_used):
        if engine.organisms[org_idx].alive != 1:
            continue
        org_id = org_idx + 1

        wants_grow = bool(engine.organisms[org_idx].brain_wants_grow)
        wants_reproduce = bool(engine.organisms[org_idx].brain_wants_reproduce)

        if not wants_grow and not wants_reproduce:
            continue

        view = engine._build_organism_view(org_idx, grid_ct, grid_oid)
        border_empties = _find_border_empty_neighbors(
            org_id, view.cells, grid_ct, grid_oid, grid_tt,
            engine.width, engine.height,
        )

        if not border_empties:
            continue

        available_energy = int(engine.organisms[org_idx].energy)

        if wants_reproduce:
            has_reproductive = any(
                ct == int(CellType.REPRODUCTIVE) for _, _, ct in view.cells
            )
            if has_reproductive and border_empties:
                seed_ct = int(CellType.SKIN)
                seed_cost = CELL_PROPERTIES[seed_ct].growth_cost
                total_repro_cost = BASE_REPRODUCTION_COST + seed_cost + DEFAULT_OFFSPRING_ENERGY
                if available_energy >= total_repro_cost:
                    sq, sr = border_empties[0]
                    reproduction_requests.append(_ReproClaim(
                        parent_id=org_id, q=sq, r=sr,
                        seed_cell_type=seed_ct, total_cost=total_repro_cost,
                    ))
                    available_energy -= total_repro_cost
                    border_empties = border_empties[1:]

        if wants_grow and border_empties:
            body_plan = engine.get_body_plan(org_id)
            genome_data = engine.genome_registry.get(
                engine.organism_genome_map.get(org_id, 0), {},
            )
            requests = body_plan.query_growth(view, border_empties, genome_data)
            requests.sort(key=lambda r: -r.priority)

            for req in requests:
                cost = CELL_PROPERTIES[req.cell_type].growth_cost
                if cost <= available_energy:
                    growth_requests.append(_GrowthClaim(
                        org_id=org_id, q=req.q, r=req.r,
                        cell_type=req.cell_type, priority=req.priority,
                    ))
                    available_energy -= cost

    _apply_growth(engine, growth_requests, grid_oid)
    _apply_reproduction(engine, reproduction_requests)


def _find_border_empty_neighbors(
    org_id: OrganismId,
    cells: list[tuple[int, int, int]],
    grid_ct: np.ndarray,
    grid_oid: np.ndarray,
    grid_tt: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for cq, cr, _ in cells:
        for nq, nr in neighbors(cq, cr, width, height):
            if (nq, nr) in seen:
                continue
            seen.add((nq, nr))
            nidx = nr * width + nq
            if grid_ct[nidx] == CellType.EMPTY and grid_oid[nidx] == 0:
                terrain = int(grid_tt[nidx])
                if terrain != int(TerrainType.ROCK):
                    result.append((nq, nr))

    return result


class _ResolvedClaim(NamedTuple):
    org_id: OrganismId
    cell_type: int
    grow_priority: int


def _apply_growth(
    engine: SimulationEngine,
    requests: list[_GrowthClaim],
    grid_oid: np.ndarray,
) -> None:
    if not requests:
        return

    claimed: dict[tuple[int, int], _ResolvedClaim] = {}

    for req in requests:
        org_idx = req.org_id - 1
        cell_count = int(engine.organisms[org_idx].cell_count)
        grow_priority = (
            (MAX_ORGANISM_CELLS - cell_count) * MID_PRIORITY_SCALE
            + req.org_id
        )

        key = (req.q, req.r)
        if key not in claimed or grow_priority > claimed[key].grow_priority:
            claimed[key] = _ResolvedClaim(
                org_id=req.org_id, cell_type=req.cell_type,
                grow_priority=grow_priority,
            )

    for (q, r), claim in claimed.items():
        idx = r * engine.width + q
        current_ct = int(engine.grid[idx].cell_type)
        current_oid = int(engine.grid[idx].organism_id)
        if current_ct != CellType.EMPTY or current_oid != 0:
            continue

        engine.grid[idx].cell_type = claim.cell_type
        engine.grid[idx].organism_id = claim.org_id

        cost = CELL_PROPERTIES[claim.cell_type].growth_cost
        org_idx = claim.org_id - 1
        engine.organisms[org_idx].energy -= cost


def _apply_reproduction(
    engine: SimulationEngine,
    requests: list[_ReproClaim],
) -> None:
    for req in requests:
        idx = req.r * engine.width + req.q
        current_ct = int(engine.grid[idx].cell_type)
        current_oid = int(engine.grid[idx].organism_id)
        if current_ct != CellType.EMPTY or current_oid != 0:
            continue

        parent_idx = req.parent_id - 1
        parent_genome = engine.organism_genome_map.get(req.parent_id, 0)
        parent_brain = engine.get_brain(req.parent_id)
        parent_body_plan = engine.get_body_plan(req.parent_id)

        child_id = engine.create_organism(
            seed_q=req.q, seed_r=req.r,
            seed_cell_type=req.seed_cell_type,
            starting_energy=DEFAULT_OFFSPRING_ENERGY,
            genome_id=parent_genome,
            brain=parent_brain,
            body_plan=parent_body_plan,
        )

        engine.organisms[parent_idx].energy -= req.total_cost

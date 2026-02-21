from __future__ import annotations
import taichi as ti

from typing import TYPE_CHECKING

from simulator.hex_grid import neighbor_offset
from simulator.types import ALIVE, DEAD

if TYPE_CHECKING:
    from simulator.engine import GridSpecs, Organism


@ti.func
def execute_movement(
        grid: ti.template(),
        temp_grid: ti.template(),
        organisms: ti.template(),
        next_org_id: ti.u16,
        grid_specs: GridSpecs,
) -> None:

    # ------------------------------------------------------------------------
    # ADD TICK MOVEMENT POINTS TODO: rename to speed_points ???
    # ------------------------------------------------------------------------
    for oid in range(next_org_id):
        if organisms[oid].alive == ALIVE and organisms[oid].locomotion_power > 0:
            organisms[oid].movement_points += organisms[oid].locomotion_power
            # Cap at points to move total_mass to prevent banking movement
            if organisms[oid].movement_points > organisms[oid].total_mass:
                organisms[oid].movement_points = organisms[oid].total_mass


    # ------------------------------------------------------------------------
    # MOVEMENT ELIGIBILITY CHECKS:
    # ------------------------------------------------------------------------
    TOP_SCALE = ti.cast(1_000_000_000, ti.i64)
    MAX_ORG_CELLS = ti.cast(1_000_000, ti.i64)

    for oid in range(next_org_id): # TODO: move into top loop for efficiency, no dependency on top loop completing
        organisms[oid].can_move = 1 # assume it can move

        if organisms[oid].alive == DEAD:
            organisms[oid].can_move = 0
        if organisms[oid].brain_move_dir < 0: # no movement desired
            organisms[oid].can_move = 0
        if organisms[oid].movement_points < organisms[oid].total_mass:
            organisms[oid].can_move = 0

        if organisms[oid].energy < get_movement_energy_cost(organisms[oid]):
            organisms[oid].can_move = 0


        # CALCULATE MOVEMENT PRIORITY
        if organisms[oid].can_move == 0:
            # not moving organisms have largest priority to keep occupying their current cells
            # TODO duplicated logic below
            priority = ti.static(2**64 - 1)
        else:
            speed_score = organisms[oid].locomotion_power // organisms[oid].total_mass
            priority = (
                    # Speed Score is highest priority
                    ti.cast(speed_score, ti.u64) * TOP_SCALE
                    # Break Ties with LOWER cell count
                    + ti.cast(MAX_ORG_CELLS - organisms[oid].cell_count, ti.u64)
                    # Final Tire Breaker guaranteed to not conflict, organism_id (same as index) larger (younger) wins
                    + ti.cast(oid, ti.u64)
            )

        organisms[oid].movement_priority = priority

    # ------------------------------------------------------------------------
    # Validate Movement is Valid + Write Claims (per grid cell)
    # ------------------------------------------------------------------------
    # TODO: TOP LEVEL LOOP
    claims = ti.field(dtype=ti.i64, shape=grid_specs.size)

    for idx in range(grid_specs.size):
        oid = grid[idx].organism_id
        if oid == 0:
            continue

        if organisms[oid].can_move == 0:
            ti.atomic_max(claims[idx], ti.static(2**64 - 1)) # not moving = max priority
            continue

        # TODO logic duplicated below
        q = idx % grid_specs.width
        r = idx // grid_specs.width
        d = organisms[oid].brain_move_dir
        nq = (q + neighbor_offset(d)[d][0]) % grid_specs.width
        nr = (r + neighbor_offset(d)[d][1]) % grid_specs.height
        dest = nr * grid_specs.width + nq

        # --- Empty + Dead Cell Check ---
        # TODO: maybe could be reworked to include dead/non-organism cells in priority calculations
        # if cell has no org_id BUT has a type, its a dead cell, cannot move there
        # if grid[dest].organism_id == 0 & grid[dest].cell_type != 0:
        # TODO: missed edge case of organism failing to due to priority conflict, just assume you cannot move into a currently
        # occupied cell for simplicity
        dest_oid = grid[dest].organism_id
        if grid[dest].cell_type != 0 & dest_oid != oid: # can only move into empty cells or its own cells
            organisms[oid].can_move = 0
            continue

        ti.atomic_max(claims[dest], organisms[oid].priority)

    # ------------------------------------------------------------------------
    # Verify Claims
    # ------------------------------------------------------------------------
    for idx in range(grid_specs.size):
        oid = grid[idx].organism_id
        if oid == 0:
            continue

        if organisms[oid].can_move == 0:
            continue

        # TODO logic duplicated above
        q = idx % grid_specs.width
        r = idx // grid_specs.width
        d = organisms[oid].brain_move_dir
        nq = (q + neighbor_offset(d)[d][0]) % grid_specs.width
        nr = (r + neighbor_offset(d)[d][1]) % grid_specs.height
        dest = nr * grid_specs.width + nq

        dest_oid = grid[dest].organism_id
        if dest_oid == oid:
            continue  # self-overlap, always ok TODO: is this check redundant? priority is unique per organism

        if claims[dest] != organisms[oid].priority: # priority conflict b/c priority is unique per organism
            organisms[oid].can_move = 0


    # ------------------------------------------------------------------------
    # Execute Moves
    # ------------------------------------------------------------------------

    # Copy current state
    for idx in range(grid_specs.size):
        temp_grid[idx] = grid[idx]

    # Clear source cells of movers
    for idx in range(grid_specs.size):
        oid = grid[idx].organism_id
        if oid > 0 and organisms[oid].can_move == 1:
            temp_grid[idx].organism_id = 0
            temp_grid[idx].cell_type = 0

    # Write destination cells
    for idx in range(grid_specs.size):
        oid = grid[idx].organism_id
        if oid > 0 and organisms[oid].can_move == 1:
            # TODO logic duplicated above
            q = idx % grid_specs.width
            r = idx // grid_specs.width
            d = organisms[oid].brain_move_dir
            nq = (q + neighbor_offset(d)[d][0]) % grid_specs.width
            nr = (r + neighbor_offset(d)[d][1]) % grid_specs.height
            dest = nr * grid_specs.width + nq

            temp_grid[dest].cell_type = grid[idx].cell_type
            temp_grid[dest].organism_id = oid

    # Commit
    for idx in range(grid_specs.size):
        grid[idx].cell_type = temp_grid[idx].cell_type
        grid[idx].organism_id = temp_grid[idx].organism_id

    # Deduct Costs
    for oid in range(next_org_id):
        if organisms[oid].can_move == 1:
            organisms[oid].movement_points -= organisms[oid].total_mass
            organisms[oid].energy -= get_movement_energy_cost(organisms[oid])




@ti.func
def get_movement_energy_cost(organism: Organism) -> ti.i32:
    return organism.total_mass # TODO: consider making this a multiplier / allowing fractional mass costs

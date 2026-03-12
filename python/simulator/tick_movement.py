import taichi as ti
from simulator.hex_grid import neighbor_offset
from simulator.sim_types import ALIVE, DEAD


# ------------------------------------------------------------------------
# ADD TICK MOVEMENT POINTS
# COMPUTE MOVEMENT ELIGIBILITY
# CALCULATE MOVEMENT PRIORITY
# ------------------------------------------------------------------------
@ti.func
def compute_movers_and_priorities(
        oid,
        organisms,
):
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

    organisms[oid].can_move = ti.cast(1, ti.i8)  # assume it can move

    if organisms[oid].alive == DEAD:
        organisms[oid].can_move = ti.cast(0, ti.i8)
    if organisms[oid].brain_move_dir < 0:  # no movement desired
        organisms[oid].can_move = ti.cast(0, ti.i8)
    if organisms[oid].movement_points < organisms[oid].total_mass:
        organisms[oid].can_move = ti.cast(0, ti.i8)

    if organisms[oid].energy < get_movement_energy_cost(organisms[oid]):
        organisms[oid].can_move = ti.cast(0, ti.i8)


    # CALCULATE MOVEMENT PRIORITY
    priority = ti.u64(0)
    if organisms[oid].can_move == 0:
        # not moving organisms have largest priority to keep occupying their current cells
        # TODO duplicated logic below
        priority = ti.u64(0xFFFFFFFFFFFFFFFF)
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
@ti.func
def write_claims(
        idx,
        organisms,
        grid,
        width,
        height,
        grid_size,
        claims,
):
    oid = grid[idx].organism_id
    if oid > 0:
        if organisms[oid].can_move == 0:
            ti.atomic_max(claims[idx], ti.u64(0xFFFFFFFFFFFFFFFF))
        else:
            q = idx % width
            r = idx // width
            d = organisms[oid].brain_move_dir
            nq = (q + neighbor_offset(d)[0]) % width
            nr = (r + neighbor_offset(d)[1]) % height
            dest = nr * width + nq

            # --- Empty + Dead Cell Check ---
            # TODO: maybe could be reworked to include dead/non-organism cells in priority calculations
            # if cell has no org_id BUT has a type, its a dead cell, cannot move there
            # if grid[dest].organism_id == 0 & grid[dest].cell_type != 0:
            # TODO: missed edge case of organism failing to due to priority conflict, just assume you cannot move into a currently
            # occupied cell for simplicity
            dest_oid = grid[dest].organism_id
            if grid[dest].cell_type != 0 and dest_oid != oid:
                organisms[oid].can_move = ti.cast(0, ti.i8)
            else:
                ti.atomic_max(claims[dest], organisms[oid].movement_priority)


# ------------------------------------------------------------------------
# Verify Claims + Invalidate Lower Priority Claims
# ------------------------------------------------------------------------
@ti.func
def invalidate_conflicting_claims(
        idx,
        organisms,
        grid,
        width,
        height,
        grid_size,
        claims,
):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].can_move == 1:
        q = idx % width
        r = idx // width
        d = organisms[oid].brain_move_dir
        nq = (q + neighbor_offset(d)[0]) % width
        nr = (r + neighbor_offset(d)[1]) % height
        dest = nr * width + nq

        dest_oid = grid[dest].organism_id
        if dest_oid != oid: # TODO: is this check redundant? priority is unique per organism
            if claims[dest] != organisms[oid].movement_priority:
                organisms[oid].can_move = ti.cast(0, ti.i8)


# ------------------------------------------------------------------------
# Copy to Temp Grid to prep for movement
# ------------------------------------------------------------------------
@ti.func
def copy_grid_to_temp(
        idx,
        grid,
        temp_grid,
):
    temp_grid[idx] = grid[idx]



# ------------------------------------------------------------------------
# Clear source cells of movers
# ------------------------------------------------------------------------
@ti.func
def clear_mover_source_cells(
        idx,
        grid,
        organisms,
        temp_grid,
):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].can_move == 1:
        temp_grid[idx].organism_id = 0
        temp_grid[idx].cell_type = ti.cast(0, ti.i8)
        temp_grid[idx].direction = ti.cast(0, ti.i8)


# ------------------------------------------------------------------------
# Write destination cells for movers
# ------------------------------------------------------------------------
@ti.func
def write_mover_destination_cells(
        idx,
        grid,
        organisms,
        width,
        height,
        grid_size,
        temp_grid,
):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].can_move == 1:
        q = idx % width
        r = idx // width
        d = organisms[oid].brain_move_dir
        nq = (q + neighbor_offset(d)[0]) % width
        nr = (r + neighbor_offset(d)[1]) % height
        dest = nr * width + nq

        temp_grid[dest].cell_type = grid[idx].cell_type
        temp_grid[dest].organism_id = oid
        temp_grid[dest].direction = grid[idx].direction


# ------------------------------------------------------------------------
# Commit temp grid back to grid
# ------------------------------------------------------------------------
@ti.func
def commit_temp_grid(
        idx,
        grid,
        temp_grid,
):
    grid[idx].cell_type = temp_grid[idx].cell_type
    grid[idx].organism_id = temp_grid[idx].organism_id
    grid[idx].direction = temp_grid[idx].direction


# ------------------------------------------------------------------------
# Deduct movement costs from movers
# ------------------------------------------------------------------------
@ti.func
def deduct_movement_costs(
        oid,
        organisms,
):
    if organisms[oid].can_move == 1:
        organisms[oid].movement_points -= organisms[oid].total_mass
        organisms[oid].energy -= get_movement_energy_cost(organisms[oid])




@ti.func
def get_movement_energy_cost(organism) -> ti.i32:
    return ti.max(1, organism.cell_count // 3)

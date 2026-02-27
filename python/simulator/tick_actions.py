import taichi as ti
from simulator.hex_grid import neighbor_offset
from simulator.sim_types import ALIVE

GROWTH_RANK: int = 1
REPRODUCE_RANK: int = 2
DESTROY_RANK: int = 3
EAT_RANK: int = 4

# claim = action_rank * 10^11 + (10^6 - cell_count) * 10^5 + org_id
# Scale factors inline as ti.i64() in each @ti.func (Taichi can't use module-level i64 constants)


@ti.func
def encode_action_claim(action_rank: ti.i32, cell_count: ti.i32, org_id: ti.i32) -> ti.i64:
    rank_scale = ti.i64(100_000_000_000)
    org_scale = ti.i64(100_000)
    max_cells = ti.i64(1_000_000)
    return (
        ti.cast(action_rank, ti.i64) * rank_scale
        + (max_cells - ti.cast(cell_count, ti.i64)) * org_scale
        + ti.cast(org_id, ti.i64)
    )


@ti.func
def init_action_state(oid: ti.i32, organisms: ti.template()):
    if organisms[oid].alive == ALIVE:
        organisms[oid].growth_cells_placed = 0
        organisms[oid].reproduce_cells_placed = 0


@ti.func
def write_action_claims(
    idx: ti.i32,
    grid: ti.template(),
    organisms: ti.template(),
    ct_action_rank: ti.template(),
    ct_can_eat: ti.template(),
    ct_can_destroy: ti.template(),
    ct_can_reproduce: ti.template(),
    ct_growth_cost: ti.template(),
    action_claims: ti.template(),
    width: ti.i32,
    height: ti.i32,
):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].alive == ALIVE:
        ct = ti.cast(grid[idx].cell_type, ti.i32)
        q = idx % width
        r = idx // width
        my_genome = organisms[oid].genome_id
        cell_count = organisms[oid].cell_count
        rank = ct_action_rank[ct]

        # --- Eat claims (rank 4): first-valid-wins per cell ---
        if rank >= EAT_RANK:
            ate = 0
            for d in ti.static(range(6)):
                if ate == 0:
                    off = neighbor_offset(d)
                    nq = (q + off[0]) % width
                    nr = (r + off[1]) % height
                    nidx = nr * width + nq
                    target_ct = ti.cast(grid[nidx].cell_type, ti.i32)
                    target_oid = grid[nidx].organism_id
                    if target_ct > 0 and ct_can_eat[ct, target_ct] > 0:
                        is_edible = 0
                        if target_oid == 0:
                            is_edible = 1
                        elif organisms[target_oid].genome_id != my_genome:
                            is_edible = 1
                        if is_edible == 1:
                            claim = encode_action_claim(EAT_RANK, cell_count, oid)
                            ti.atomic_max(action_claims[nidx], claim)
                            ate = 1

        # --- Destroy claims (rank 3): ALL adjacent foreign cells ---
        if rank == DESTROY_RANK:
            for d in ti.static(range(6)):
                off = neighbor_offset(d)
                nq = (q + off[0]) % width
                nr = (r + off[1]) % height
                nidx = nr * width + nq
                target_ct = ti.cast(grid[nidx].cell_type, ti.i32)
                target_oid = grid[nidx].organism_id
                if target_ct > 0 and target_oid > 0 and ct_can_destroy[ct, target_ct] > 0:
                    if organisms[target_oid].genome_id != my_genome:
                        claim = encode_action_claim(DESTROY_RANK, cell_count, oid)
                        ti.atomic_max(action_claims[nidx], claim)

        # --- Reproduce claims (rank 2): brain-directed, one per org ---
        if organisms[oid].brain_reproduce_cell_idx == idx and ct_can_reproduce[ct] == 1:
            rep_dir = organisms[oid].brain_reproduce_direction
            off = neighbor_offset(rep_dir)
            nq = (q + off[0]) % width
            nr = (r + off[1]) % height
            nidx = nr * width + nq
            if grid[nidx].cell_type == 0 and grid[nidx].organism_id == 0:
                organisms[oid].brain_reproduce_cell_type = ct
                claim = encode_action_claim(REPRODUCE_RANK, cell_count, oid)
                ti.atomic_max(action_claims[nidx], claim)

        # --- Growth claims (rank 1): border cells of growing organisms ---
        if organisms[oid].brain_wants_grow == 1:
            grow_type = organisms[oid].brain_grow_cell_type
            cost = ct_growth_cost[grow_type]
            if organisms[oid].energy >= cost:
                grow_dir = organisms[oid].brain_grow_direction
                for d in ti.static(range(6)):
                    dir_ok = 0
                    if grow_dir < 0:
                        dir_ok = 1
                    elif d == grow_dir:
                        dir_ok = 1
                    if dir_ok == 1:
                        off = neighbor_offset(d)
                        nq = (q + off[0]) % width
                        nr = (r + off[1]) % height
                        nidx = nr * width + nq
                        if grid[nidx].cell_type == 0 and grid[nidx].organism_id == 0:
                            claim = encode_action_claim(GROWTH_RANK, cell_count, oid)
                            ti.atomic_max(action_claims[nidx], claim)


@ti.func
def execute_action_claims(
    idx: ti.i32,
    grid: ti.template(),
    organisms: ti.template(),
    ct_consumption_value: ti.template(),
    ct_growth_cost: ti.template(),
    action_claims: ti.template(),
    reproduce_buffer: ti.template(),
):
    claim = action_claims[idx]
    if claim > 0:
        rank_scale = ti.i64(100_000_000_000)
        org_scale = ti.i64(100_000)
        claimed_rank = claim // rank_scale
        actor_oid = ti.cast(claim % org_scale, ti.i32)

        if claimed_rank >= EAT_RANK:
            # EAT: destroy cell + gain energy
            target_ct = ti.cast(grid[idx].cell_type, ti.i32)
            if target_ct > 0:
                target_oid = grid[idx].organism_id
                energy_gained = ct_consumption_value[target_ct]
                ti.atomic_add(organisms[actor_oid].energy, energy_gained)
                if target_oid > 0:
                    organisms[target_oid].needs_connectivity_check = ti.cast(1, ti.i8)
                grid[idx].cell_type = ti.cast(0, ti.i8)
                grid[idx].organism_id = 0
                grid[idx].direction = ti.cast(0, ti.i8)

        elif claimed_rank == DESTROY_RANK:
            # DESTROY: destroy cell, no energy gain
            target_ct = ti.cast(grid[idx].cell_type, ti.i32)
            if target_ct > 0:
                target_oid = grid[idx].organism_id
                if target_oid > 0:
                    organisms[target_oid].needs_connectivity_check = ti.cast(1, ti.i8)
                grid[idx].cell_type = ti.cast(0, ti.i8)
                grid[idx].organism_id = 0
                grid[idx].direction = ti.cast(0, ti.i8)

        elif claimed_rank == REPRODUCE_RANK:
            # REPRODUCE: deduct cost, write parent_oid to reproduce_buffer
            prev = ti.atomic_add(organisms[actor_oid].reproduce_cells_placed, 1)
            if prev < 1:
                seed_ct = organisms[actor_oid].brain_reproduce_cell_type
                rep_energy = organisms[actor_oid].brain_reproduce_energy
                total_cost = ct_growth_cost[seed_ct] + rep_energy
                if organisms[actor_oid].energy >= total_cost:
                    ti.atomic_sub(organisms[actor_oid].energy, total_cost)
                    reproduce_buffer[idx] = actor_oid

        elif claimed_rank == GROWTH_RANK:
            # GROWTH: place new cell for existing organism
            prev = ti.atomic_add(organisms[actor_oid].growth_cells_placed, 1)
            if prev < 1:
                grow_type = organisms[actor_oid].brain_grow_cell_type
                cost = ct_growth_cost[grow_type]
                if organisms[actor_oid].energy >= cost:
                    ti.atomic_sub(organisms[actor_oid].energy, cost)
                    grid[idx].cell_type = ti.cast(grow_type, ti.i8)
                    grid[idx].organism_id = actor_oid

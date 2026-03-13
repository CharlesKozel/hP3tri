import taichi as ti
from simulator.hex_grid import neighbor_offset
from simulator.cell_types import NUM_CELL_TYPES, CellType

SENSOR_NOTHING: int = 9999
FOOD_TYPE: int = int(CellType.FOOD)

_CH_FOOD: int = 0
_CH_FOREIGN_ALIVE: int = 1
_CH_DEAD: int = 2
_CH_DANGEROUS: int = 3
_CH_FRIENDLY: int = 4
_CH_OPEN_SPACE: int = 5
_CH_EDIBLE: int = 6
_NUM_CHANNELS: int = 7
_NUM_SECTORS: int = 6


@ti.func
def init_sensor_organism(oid, sensor_distances):
    for s in ti.static(range(_NUM_SECTORS)):
        for c in ti.static(range(_NUM_CHANNELS)):
            sensor_distances[oid, s, c] = SENSOR_NOTHING
        sensor_distances[oid, s, _CH_OPEN_SPACE] = 0


@ti.func
def classify_and_record(target_ct, target_oid, scanner_oid, direction, distance,
                        organisms, can_eat, can_destroy,
                        sensor_distances):
    my_genome = organisms[scanner_oid].genome_id

    if target_ct == 0 and target_oid == 0:
        ti.atomic_max(sensor_distances[scanner_oid, direction, _CH_OPEN_SPACE], 1)
    else:
        is_food = 0
        if target_ct == FOOD_TYPE and target_oid == 0:
            ti.atomic_min(sensor_distances[scanner_oid, direction, _CH_FOOD], distance)
            is_food = 1

        if target_oid == 0 and target_ct > 0 and is_food == 0:
            ti.atomic_min(sensor_distances[scanner_oid, direction, _CH_DEAD], distance)

        is_friendly = 0
        if target_oid > 0:
            target_genome = organisms[target_oid].genome_id
            if target_genome == my_genome:
                ti.atomic_min(sensor_distances[scanner_oid, direction, _CH_FRIENDLY], distance)
                is_friendly = 1
            else:
                ti.atomic_min(sensor_distances[scanner_oid, direction, _CH_FOREIGN_ALIVE], distance)
                dangerous = 0
                for ct in ti.static(range(NUM_CELL_TYPES)):
                    if can_eat[target_ct, ct] > 0:
                        dangerous = 1
                    if can_destroy[target_ct, ct] > 0:
                        dangerous = 1
                if dangerous == 1:
                    ti.atomic_min(sensor_distances[scanner_oid, direction, _CH_DANGEROUS], distance)

        if is_friendly == 0 and target_ct > 0:
            edible = 0
            for eater_ct in ti.static(range(NUM_CELL_TYPES)):
                if organisms[scanner_oid].cell_type_counts[eater_ct] > 0:
                    if can_eat[eater_ct, target_ct] > 0:
                        edible = 1
            if edible == 1:
                ti.atomic_min(sensor_distances[scanner_oid, direction, _CH_EDIBLE], distance)


@ti.func
def scan_ray(start_q, start_r, direction, v_range, oid,
             grid, organisms, can_eat, can_destroy, sensor_distances,
             width, height):
    cq = start_q
    cr = start_r
    found = 0
    for step in range(v_range):
        if found == 0:
            off = neighbor_offset(direction)
            cq = (cq + off[0]) % width
            cr = (cr + off[1]) % height
            cidx = cr * width + cq
            target_oid = grid[cidx].organism_id
            target_ct = ti.cast(grid[cidx].cell_type, ti.i32)
            is_own = 0
            if target_oid == oid:
                is_own = 1
            is_empty = 0
            if target_ct == 0 and target_oid == 0:
                is_empty = 1
            if is_own == 0 and is_empty == 0:
                classify_and_record(
                    target_ct, target_oid, oid, direction, step + 1,
                    organisms, can_eat, can_destroy, sensor_distances,
                )
                found = 1
            if is_empty == 1:
                classify_and_record(
                    target_ct, target_oid, oid, direction, step + 1,
                    organisms, can_eat, can_destroy, sensor_distances,
                )


@ti.func
def scan_hex_target(q, r, oid, sector, distance,
                    grid, organisms, can_eat, can_destroy, sensor_distances,
                    width, height):
    wq = q % width
    wr = r % height
    cidx = wr * width + wq
    target_oid = grid[cidx].organism_id
    target_ct = ti.cast(grid[cidx].cell_type, ti.i32)
    if target_oid != oid:
        classify_and_record(
            target_ct, target_oid, oid, sector, distance,
            organisms, can_eat, can_destroy, sensor_distances,
        )


@ti.func
def scan_cone(start_q, start_r, primary_dir, v_range, v_expansion, oid,
              grid, organisms, can_eat, can_destroy, sensor_distances,
              width, height):
    primary_off = neighbor_offset(primary_dir)
    left_dir = (primary_dir + 5) % 6
    right_dir = (primary_dir + 1) % 6
    left_off = neighbor_offset(left_dir)
    right_off = neighbor_offset(right_dir)

    for d in range(v_range):
        step = d + 1
        cq = (start_q + primary_off[0] * step) % width
        cr = (start_r + primary_off[1] * step) % height

        scan_hex_target(cq, cr, oid, primary_dir, step,
                        grid, organisms, can_eat, can_destroy, sensor_distances,
                        width, height)

        spread = 0
        if v_expansion > 0:
            spread = step // v_expansion

        for lat in range(spread):
            offset = lat + 1
            lq = (cq + left_off[0] * offset) % width
            lr = (cr + left_off[1] * offset) % height
            scan_hex_target(lq, lr, oid, primary_dir, step,
                            grid, organisms, can_eat, can_destroy, sensor_distances,
                            width, height)

            rq = (cq + right_off[0] * offset) % width
            rr = (cr + right_off[1] * offset) % height
            scan_hex_target(rq, rr, oid, primary_dir, step,
                            grid, organisms, can_eat, can_destroy, sensor_distances,
                            width, height)


@ti.func
def cell_vision(idx, grid, organisms,
                ct_vision_range, ct_vision_expansion, ct_directional,
                can_eat, can_destroy, sensor_distances, width, height):
    oid = grid[idx].organism_id
    ct = ti.cast(grid[idx].cell_type, ti.i32)
    v_range = ct_vision_range[ct]
    if v_range > 0:
        q = idx % width
        r = idx // width
        if ct_directional[ct] == 1:
            primary_dir = ti.cast(grid[idx].direction, ti.i32)
            v_expansion = ct_vision_expansion[ct]
            scan_cone(q, r, primary_dir, v_range, v_expansion, oid,
                      grid, organisms, can_eat, can_destroy, sensor_distances,
                      width, height)
        else:
            for d in ti.static(range(6)):
                scan_ray(q, r, d, v_range, oid,
                         grid, organisms, can_eat, can_destroy, sensor_distances,
                         width, height)
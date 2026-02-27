import taichi as ti
from simulator.hex_grid import neighbor_offset
from simulator.sim_types import ALIVE

SENTINEL_LABEL: int = 0x7FFFFFFF


@ti.func
def mark_organism_death(oid, organisms):
    if organisms[oid].alive == ALIVE:
        if organisms[oid].cell_count == 0 or organisms[oid].energy <= 0:
            organisms[oid].alive = 0


@ti.func
def clear_dead_cells(idx, grid, organisms):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].alive == 0:
        grid[idx].organism_id = 0


@ti.func
def init_connectivity_labels(idx, grid, organisms, labels):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].needs_connectivity_check == 1:
        labels[idx] = idx
    else:
        labels[idx] = SENTINEL_LABEL


@ti.func
def propagate_connectivity(idx, grid, organisms, labels, changed, width, height):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].needs_connectivity_check == 1:
        my_label = labels[idx]
        q = idx % width
        r = idx // width
        min_label = my_label
        for d in ti.static(range(6)):
            off = neighbor_offset(d)
            nq = (q + off[0]) % width
            nr = (r + off[1]) % height
            nidx = nr * width + nq
            if grid[nidx].organism_id == oid:
                nl = labels[nidx]
                if nl < min_label:
                    min_label = nl
        if min_label < my_label:
            labels[idx] = min_label
            changed[None] = 1


@ti.func
def count_components(idx, grid, organisms, labels, component_size):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].needs_connectivity_check == 1:
        ti.atomic_add(component_size[labels[idx]], 1)


@ti.func
def find_best_component(idx, grid, organisms, labels, component_size, best_component):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].needs_connectivity_check == 1:
        label = labels[idx]
        size = component_size[label]
        encoded = (ti.cast(size, ti.i64) << 32) | ti.cast(label, ti.i64)
        ti.atomic_max(best_component[oid], encoded)


@ti.func
def remove_disconnected_cells(idx, grid, organisms, labels, best_component):
    oid = grid[idx].organism_id
    if oid > 0 and organisms[oid].needs_connectivity_check == 1:
        best_label = ti.cast(best_component[oid] & ti.i64(0xFFFFFFFF), ti.i32)
        if labels[idx] != best_label:
            grid[idx].organism_id = 0


@ti.func
def clear_connectivity_flags(oid, organisms):
    organisms[oid].needs_connectivity_check = ti.cast(0, ti.i8)

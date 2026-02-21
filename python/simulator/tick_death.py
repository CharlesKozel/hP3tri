from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from simulator.hex_grid import neighbors

if TYPE_CHECKING:
    from simulator.engine import SimulationEngine


def execute_death(engine: SimulationEngine) -> None:
    _check_starvation(engine)
    _mark_dead_cells(engine)
    _check_connectivity(engine)


def _check_starvation(engine: SimulationEngine) -> None:
    for org_idx in range(engine.organism_manager.highest_id_used):
        if engine.organisms[org_idx].alive == 1 and engine.organisms[org_idx].energy <= 0:
            engine.organisms[org_idx].alive = 0


def _mark_dead_cells(engine: SimulationEngine) -> None:
    engine._kernel_mark_dead_cells(
        engine.grid,
        engine.organisms,
        engine.grid_size,
    )


def _check_connectivity(engine: SimulationEngine) -> None:
    if not engine._organisms_took_damage:
        return

    grid_oid = engine.grid.organism_id.to_numpy()
    width = engine.width
    height = engine.height

    for org_id in list(engine._organisms_took_damage):
        org_idx = org_id - 1
        if engine.organisms[org_idx].alive != 1:
            continue

        cell_positions: list[tuple[int, int]] = []
        for idx in range(engine.grid_size):
            if grid_oid[idx] == org_id:
                q = idx % width
                r = idx // width
                cell_positions.append((q, r))

        if len(cell_positions) <= 1:
            continue

        all_cells = set(cell_positions)
        components: list[set[tuple[int, int]]] = []
        remaining = set(all_cells)

        while remaining:
            start = next(iter(remaining))
            component: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque([start])
            component.add(start)

            while queue:
                cq, cr = queue.popleft()
                for nq, nr in neighbors(cq, cr, width, height):
                    if (nq, nr) in remaining and (nq, nr) not in component:
                        component.add((nq, nr))
                        queue.append((nq, nr))

            components.append(component)
            remaining -= component

        if len(components) <= 1:
            continue

        largest = max(components, key=len)
        for comp in components:
            if comp is not largest:
                for cq, cr in comp:
                    cidx = cr * width + cq
                    engine.grid[cidx].organism_id = 0

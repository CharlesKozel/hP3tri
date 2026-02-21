from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulator.engine import SimulationEngine


def execute_actions(engine: SimulationEngine) -> None:
    engine._kernel_resolve_actions(
        engine.grid,
        engine.organisms,
        engine.ct_fields.consumption_value,
        engine.width,
        engine.height,
        engine.grid_size,
    )
    _track_damage(engine)


def _track_damage(engine: SimulationEngine) -> None:
    grid_oid = engine.grid.organism_id.to_numpy()
    for org_idx in range(engine.organism_manager.highest_id_used):
        if engine.organisms[org_idx].alive != 1:
            continue
        org_id = org_idx + 1
        old_count = int(engine.organisms[org_idx].cell_count)
        new_count = int(np.sum(grid_oid == org_id))
        if new_count < old_count:
            engine._organisms_took_damage.add(org_id)

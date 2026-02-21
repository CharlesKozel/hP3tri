from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

OrganismId: TypeAlias = int
GenomeId: TypeAlias = int
GridIndex: TypeAlias = int
GenomeData: TypeAlias = dict[str, Any]

ALIVE: int = 1
DEAD: int = 0

@dataclass
class HexGridState:
    """Snapshot of hex grid per-tile data.

    All arrays are flat with length width * height,
    indexed by grid_index = r * width + q.
    """
    width: int
    height: int
    cell_type: NDArray[np.int8]
    organism_id: NDArray[np.int32]
    terrain_type: NDArray[np.int8]
    organism_genome_map: dict[OrganismId, GenomeId]

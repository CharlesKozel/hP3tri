from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from simulator.cell_types import NUM_CELL_TYPES

NUM_SECTORS: int = 6

CH_FOOD: int = 0
CH_FOREIGN_ALIVE: int = 1
CH_DEAD: int = 2
CH_DANGEROUS: int = 3
CH_FRIENDLY: int = 4
CH_OPEN_SPACE: int = 5
CH_EDIBLE: int = 6
NUM_CHANNELS: int = 7

SENSOR_INPUT_SIZE: int = NUM_SECTORS * NUM_CHANNELS + 2 + NUM_CELL_TYPES


@dataclass
class OrganismView:
    organism_id: int
    age: int
    energy: int
    cell_count: int
    total_mass: int
    locomotion_power: int
    cells: list[tuple[int, int, int]]  # (q, r, cell_type)


def _default_sector_data() -> NDArray[np.float32]:
    data = np.ones((NUM_SECTORS, NUM_CHANNELS), dtype=np.float32)
    data[:, CH_OPEN_SPACE] = 0.0
    return data


@dataclass
class SensorInputs:
    sector_data: NDArray[np.float32] = field(default_factory=_default_sector_data)
    own_energy: float = 0.0
    own_age: float = 0.0
    own_cell_counts: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(NUM_CELL_TYPES, dtype=np.int32),
    )

    def to_flat(self) -> NDArray[np.float32]:
        return np.concatenate([
            self.sector_data.ravel(),
            np.array([self.own_energy, self.own_age], dtype=np.float32),
            self.own_cell_counts.astype(np.float32),
        ])


@dataclass
class BrainOutput:
    move_direction: int = -1  # -1 = no move, 0-5 = hex direction
    wants_grow: bool = False
    grow_direction: int = -1  # -1 = any border cell, 0-5 = specific hex direction
    grow_cell_type: int = 1   # CellType value to grow (default SOFT_TISSUE)
    reproduce_cell_idx: int = -1   # grid index of source cell, -1 = don't reproduce
    reproduce_direction: int = 0   # 0-5 hex direction for offspring placement
    reproduce_energy: int = 10     # energy given to offspring


class BrainProvider(ABC):
    @abstractmethod
    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        ...

# @ti.func
# def brain_tick(organism: Organism) -> BrainOutput:
#     # dummy brain implementation
#     # in actuality this needs to run the right brain for the genome
#     return BrainOutput(
#         move_direction = organism.age % 6
#     )
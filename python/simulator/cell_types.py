from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, TypeAlias

import taichi as ti

from simulator.hex_grid import neighbors, index
from simulator.types import OrganismId, GridIndex, GenomeId, HexGridState


# 6 minimal cell types for initial development, IMPORTANT: DONT ADD MORE
class CellType(IntEnum):
    NULL = 0 # special type to indicate removing a cell at that index + absorbing its energy
    SOFT_TISSUE = auto()
    MOUTH = auto()
    FLAGELLA = auto()
    EYE = auto()
    FOOD = auto()

    # SKIN = auto()
    # ARMOR = auto()
    # SPIKE = auto()
    # PHOTOSYNTHETIC = auto()
    # MEMBRANE = auto()
    # ROOT = auto()
    # TEETH = auto()
    # CHEMICAL_SENSOR = auto()
    # TOUCH_SENSOR = auto()
    # CILIA = auto()
    # PSEUDOPOD = auto()
    # STORAGE_VACUOLE = auto()
    # REPRODUCTIVE = auto()
    # SIGNAL_EMITTER = auto()
    # PIGMENT = auto()


NUM_CELL_TYPES = len(CellType)

@dataclass(frozen=True)
class CellProps:
    maintenance_cost: int
    growth_cost: int
    mass: int
    locomotion_power: int
    energy_generation: int
    consumption_value: int


CELL_PROPERTIES: dict[CellType, CellProps] = {
    CellType.NULL:             CellProps(0, 0,  0, 0,  0, 0),
    CellType.SOFT_TISSUE:      CellProps(1, 3,  1, 1,  0, 0),
    CellType.MOUTH:            CellProps(2, 10, 3, 0,  0, 0),
    CellType.FLAGELLA:         CellProps(2, 10, 1, 10, 0, 0),
    CellType.EYE:              CellProps(3, 8,  2, 0,  0, 0),
    CellType.FOOD:             CellProps(0, 10, 5, 0,  0, 10),

    # CellType.SKIN:             CellProps(maintenance_cost=1, growth_cost=5,  mass=2, consumption_value=3),
    # CellType.ARMOR:            CellProps(maintenance_cost=3, growth_cost=15, mass=5, consumption_value=5),
    # CellType.SPIKE:            CellProps(maintenance_cost=2, growth_cost=10, mass=3, consumption_value=4),
    # CellType.PHOTOSYNTHETIC:   CellProps(maintenance_cost=1, growth_cost=8,  mass=2, energy_generation=2, consumption_value=4),
    # CellType.MEMBRANE:         CellProps(maintenance_cost=1, growth_cost=4,  mass=1, consumption_value=2),
    # CellType.ROOT:             CellProps(maintenance_cost=1, growth_cost=6,  mass=2, energy_generation=1, consumption_value=3),
    # CellType.TEETH:            CellProps(maintenance_cost=2, growth_cost=12, mass=3, consumption_value=4),
    # CellType.CHEMICAL_SENSOR:  CellProps(maintenance_cost=1, growth_cost=6,  mass=1, consumption_value=2),
    # CellType.TOUCH_SENSOR:     CellProps(maintenance_cost=1, growth_cost=5,  mass=1, consumption_value=2),
    # CellType.CILIA:            CellProps(maintenance_cost=1, growth_cost=6,  mass=1, locomotion_power=1, consumption_value=2),
    # CellType.PSEUDOPOD:        CellProps(maintenance_cost=1, growth_cost=2,  mass=1, consumption_value=1),
    # CellType.STORAGE_VACUOLE:  CellProps(maintenance_cost=1, growth_cost=8,  mass=2, consumption_value=5),
    # CellType.REPRODUCTIVE:     CellProps(maintenance_cost=2, growth_cost=12, mass=2, consumption_value=4),
    # CellType.SIGNAL_EMITTER:   CellProps(maintenance_cost=1, growth_cost=6,  mass=1, consumption_value=2),
    # CellType.PIGMENT:          CellProps(maintenance_cost=0, growth_cost=3,  mass=1, consumption_value=1),
}

class CellActionType(IntEnum):
    DESTROY = 0    # remove cell at target, gain energy
    PLACE = 1      # place new cell at target

@dataclass(frozen=True)
class CellActionResult:
    action: CellActionType
    target_index: GridIndex = 0

CellActionFunc = Callable[[HexGridState, OrganismId, GenomeId, GridIndex],list[CellActionResult]]

def getCellActions(cell_type: CellType) -> CellActionFunc | None:
    match cell_type:
        case CellType.MOUTH:
            return mouth_action
    return None

def mouth_action(state: HexGridState, org_id: OrganismId, genome_id: GenomeId, target_index: GridIndex) -> list[CellActionResult]:
    eatable_cells = {CellType.FOOD, CellType.SOFT_TISSUE, CellType.FLAGELLA, CellType.EYE}
    q = target_index % state.width
    r = target_index // state.width
    for nq, nr in neighbors(q, r, state.width, state.height):
        nidx = index(nq, nr, state.width)
        ct = CellType(state.cell_type[nidx])
        if ct not in eatable_cells:
            continue
        n_oid = int(state.organism_id[nidx])
        if n_oid == 0 or state.organism_genome_map.get(n_oid) != genome_id:
            return [CellActionResult(action=CellActionType.DESTROY, target_index=nidx)]
    return []

class CellTypeFields:
    def __init__(self) -> None:
        self.maintenance_cost = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.growth_cost = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.mass = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.locomotion_power = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.energy_generation = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.consumption_value = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)

    def load(self) -> None:
        for ct, props in CELL_PROPERTIES.items():
            self.maintenance_cost[ct] = props.maintenance_cost
            self.growth_cost[ct] = props.growth_cost
            self.mass[ct] = props.mass
            self.locomotion_power[ct] = props.locomotion_power
            self.energy_generation[ct] = props.energy_generation
            self.consumption_value[ct] = props.consumption_value

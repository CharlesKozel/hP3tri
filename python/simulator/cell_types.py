from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, TypeAlias

import taichi as ti

from simulator.hex_grid import neighbors, index
from simulator.sim_types import OrganismId, GridIndex, GenomeId, HexGridState


class CellType(IntEnum):
    NULL = 0
    SOFT_TISSUE = auto()
    MOUTH = auto()
    FLAGELLA = auto()
    EYE = auto()
    SPIKE = auto()
    FOOD = auto()
    PHOTOSYNTHETIC = auto()
    ARMOR = auto()
    SKIN = auto()


NUM_CELL_TYPES = len(CellType)
NUM_ORGANISM_CELL_TYPES = NUM_CELL_TYPES - 2  # exclude NULL and FOOD

@dataclass(frozen=True)
class CellProps:
    maintenance_cost: int
    growth_cost: int
    mass: int
    locomotion_power: int
    energy_generation: int
    consumption_value: int
    color: str
    display_name: str
    vision_range: int = 0
    vision_expansion: int = 0
    directional: int = 0
    action_rank: int = 0
    can_reproduce: int = 0


CELL_PROPERTIES: dict[CellType, CellProps] = {
    CellType.NULL:           CellProps(0,  0,  0, 0,  0, 0,  '#000000', 'Empty'),
    CellType.SOFT_TISSUE:    CellProps(1,  2,  1, 0,  0, 1,  '#e8b4a0', 'Soft Tissue', vision_range=1, can_reproduce=1),
    CellType.MOUTH:          CellProps(2,  8,  3, 0,  0, 4,  '#cc3333', 'Mouth', vision_range=1, action_rank=4, can_reproduce=1),
    CellType.FLAGELLA:       CellProps(3,  8,  1, 10, 0, 4,  '#cc88dd', 'Flagella', vision_range=1, can_reproduce=1),
    CellType.EYE:            CellProps(2,  6,  2, 0,  0, 3,  '#ffffff', 'Eye', vision_range=5, vision_expansion=1, directional=1),
    CellType.SPIKE:          CellProps(3, 12,  4, 0,  0, 6,  '#ff6600', 'Spike', action_rank=3),
    CellType.FOOD:           CellProps(0, 10,  5, 0,  0, 10, '#66dd66', 'Food'),
    CellType.PHOTOSYNTHETIC: CellProps(1,  5,  2, 0,  3, 3,  '#33aa33', 'Photosynthetic', can_reproduce=1),
    CellType.ARMOR:          CellProps(1, 15,  6, 0,  0, 0,  '#8888aa', 'Armor'),
    CellType.SKIN:           CellProps(1,  3,  2, 0,  0, 1,  '#ddbb88', 'Skin', can_reproduce=1),
}


def get_cell_type_metadata() -> list[dict[str, object]]:
    return [
        {
            "id": int(ct),
            "name": props.display_name,
            "color": props.color,
        }
        for ct, props in CELL_PROPERTIES.items()
    ]

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

CAN_EAT: dict[CellType, set[CellType]] = {
    CellType.MOUTH: {
        CellType.FOOD, CellType.SOFT_TISSUE, CellType.MOUTH, CellType.FLAGELLA,
        CellType.EYE, CellType.SPIKE, CellType.PHOTOSYNTHETIC, CellType.SKIN,
    },
}

CAN_DESTROY: dict[CellType, set[CellType]] = {
    CellType.SPIKE: {
        CellType.SOFT_TISSUE, CellType.MOUTH, CellType.FLAGELLA, CellType.EYE,
        CellType.SPIKE, CellType.PHOTOSYNTHETIC, CellType.ARMOR, CellType.SKIN,
    },
}

def mouth_action(state: HexGridState, org_id: OrganismId, genome_id: GenomeId, target_index: GridIndex) -> list[CellActionResult]:
    raise NotImplemented()
    # THIS MUST BE IMPLEMENTED AS A TAICHI FUNCTION
    # eatable_cells = CAN_EAT.get(CellType.MOUTH, set())
    # q = target_index % state.width
    # r = target_index // state.width
    # for nq, nr in neighbors(q, r, state.width, state.height):
    #     nidx = index(nq, nr, state.width)
    #     ct = CellType(state.cell_type[nidx])
    #     if ct not in eatable_cells:
    #         continue
    #     n_oid = int(state.organism_id[nidx])
    #     if n_oid == 0 or state.organism_genome_map.get(n_oid) != genome_id:
    #         return [CellActionResult(action=CellActionType.DESTROY, target_index=nidx)]
    # return []


class CellTypeFields:
    def __init__(self) -> None:
        self.maintenance_cost = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.growth_cost = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.mass = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.locomotion_power = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.energy_generation = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.consumption_value = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.vision_range = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.vision_expansion = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.directional = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.action_rank = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)
        self.can_reproduce = ti.field(dtype=ti.i32, shape=NUM_CELL_TYPES)


        # 1 IFF self.can_eat[CELL_A, CELL_B] == 1 : A(B()), A eats B, -> 1
        # mouths, tongues, needles, fungus,
        # 'mouths/weapons' can push into things to destroy them,
        # !if conflict resolution happens!
        self.can_eat = ti.field(dtype=ti.i32, shape=(NUM_CELL_TYPES, NUM_CELL_TYPES))

        # 1 IFF self.can_destroy[CELL_A, CELL_B] == 1 : A(B()), A destroys B, -> 1
        # Spikes, rock, pressure can destroy,
        # 'mouths/weapons' can push into things to destroy them,
        # !if conflict resolution happens!
        self.can_destroy = ti.field(dtype=ti.i32, shape=(NUM_CELL_TYPES, NUM_CELL_TYPES))

        # ... expand these matrix as needed


    def load(self) -> None:
        for ct, props in CELL_PROPERTIES.items():
            self.maintenance_cost[ct] = props.maintenance_cost
            self.growth_cost[ct] = props.growth_cost
            self.mass[ct] = props.mass
            self.locomotion_power[ct] = props.locomotion_power
            self.energy_generation[ct] = props.energy_generation
            self.consumption_value[ct] = props.consumption_value
            self.vision_range[ct] = props.vision_range
            self.vision_expansion[ct] = props.vision_expansion
            self.directional[ct] = props.directional
            self.action_rank[ct] = props.action_rank
            self.can_reproduce[ct] = props.can_reproduce

        for eater, targets in CAN_EAT.items():
            for target in targets:
                self.can_eat[int(eater), int(target)] = 1
        for destroyer, targets in CAN_DESTROY.items():
            for target in targets:
                self.can_destroy[int(destroyer), int(target)] = 1

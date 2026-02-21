from abc import ABC, abstractmethod
from dataclasses import dataclass
import taichi as ti


from simulator.engine import Organism
from simulator.types import GenomeId


@dataclass
class OrganismView:
    organism_id: int
    age: int
    energy: int
    cell_count: int
    total_mass: int
    locomotion_power: int
    cells: list[tuple[int, int, int]]  # (q, r, cell_type)


@dataclass
class SensorInputs:
    nearest_food_distance: int = 0
    nearest_food_direction: int = -1
    nearest_threat_distance: int = 0
    nearest_threat_direction: int = -1
    adjacent_foreign_count: int = 0
    adjacent_food_count: int = 0


@dataclass
class BrainOutput:
    move_direction: int = -1  # -1 = no move, 0-5 = hex direction
    # wants_grow: bool = False
    # wants_reproduce: bool = False


class BrainProvider(ABC):
    @abstractmethod
    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        ...

@ti.func
def brain_tick(organism: Organism) -> BrainOutput:
    # dummy brain implementation
    # in actuality this needs to run the right brain for the genome
    return BrainOutput(
        move_direction = organism.age % 6
    )
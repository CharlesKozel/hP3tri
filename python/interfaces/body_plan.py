from abc import ABC, abstractmethod
from dataclasses import dataclass

from interfaces.brain import OrganismView


@dataclass
class GrowthRequest:
    q: int
    r: int
    cell_type: int
    priority: float = 1.0


class BodyPlanProvider(ABC):
    @abstractmethod
    def query_growth(
        self,
        organism: OrganismView,
        border_empty_neighbors: list[tuple[int, int]],
        genome_data: dict,
    ) -> list[GrowthRequest]:
        ...

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from interfaces.brain import OrganismView, SensorInputs


class SensorAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        organism: OrganismView,
        grid_cell_type: NDArray[np.int8],
        grid_organism_id: NDArray[np.int32],
        width: int,
        height: int,
    ) -> SensorInputs:
        ...

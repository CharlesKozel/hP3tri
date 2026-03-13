"""Scripted BrainProvider implementations for demo scenes."""
from __future__ import annotations

from interfaces.brain import (
    BrainOutput, BrainProvider, OrganismView, SensorInputs,
    NUM_SECTORS, CH_FOOD, CH_FOREIGN_ALIVE, CH_DANGEROUS,
    CH_OPEN_SPACE, CH_EDIBLE,
)
from simulator.cell_types import CellType, CELL_PROPERTIES


def _nearest_sector(sd, channel: int) -> tuple[int, float]:
    best_s, best_d = -1, 1.0
    for s in range(NUM_SECTORS):
        if sd[s, channel] < best_d:
            best_d = sd[s, channel]
            best_s = s
    return best_s, best_d


def _most_open_sector(sd) -> int:
    best_s, best_v = 0, -1.0
    for s in range(NUM_SECTORS):
        if sd[s, CH_OPEN_SPACE] > best_v:
            best_v = sd[s, CH_OPEN_SPACE]
            best_s = s
    return best_s


def _find_border_reproducible_cell(
    organism: OrganismView,
) -> tuple[int, int]:
    """Find a can_reproduce cell on the border with an empty neighbor.

    Returns (grid_index, direction_to_empty) or (-1, 0).
    """
    from simulator.hex_grid import NEIGHBOR_OFFSETS, wrap

    w = organism.grid_width
    h = organism.grid_height
    occupied: set[tuple[int, int]] = {(q, r) for q, r, _ in organism.cells}

    for q, r, ct in organism.cells:
        props = CELL_PROPERTIES.get(CellType(ct))
        if not (props and props.can_reproduce):
            continue
        for d, (dq, dr) in enumerate(NEIGHBOR_OFFSETS):
            nq, nr = wrap(q + dq, r + dr, w, h)
            if (nq, nr) not in occupied:
                return (r * w + q, d)
    return (-1, 0)


class PlantBrain(BrainProvider):
    """Always grow. Stages cell types by age for concentric ring appearance."""

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        age = organism.age
        if age < 30:
            ct = int(CellType.PHOTOSYNTHETIC)
        elif age < 60:
            ct = int(CellType.PHOTOSYNTHETIC)
        else:
            ct = int(CellType.SKIN)

        return BrainOutput(
            wants_grow=True,
            grow_direction=-1,
            grow_cell_type=ct,
        )


class PredatorBrain(BrainProvider):
    """Chase nearest edible organism. Can be given a target direction to seek."""

    def __init__(
        self,
        growth_ticks: int = 0,
        seek_direction: int = -1,
    ) -> None:
        self.growth_ticks = growth_ticks
        self.seek_direction = seek_direction

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        sd = sensors.sector_data

        # Growth phase (if growing from seed)
        if organism.age < self.growth_ticks:
            if organism.age < self.growth_ticks // 3:
                ct = int(CellType.SOFT_TISSUE)
            elif organism.age < 2 * self.growth_ticks // 3:
                ct = int(CellType.FLAGELLA)
            else:
                ct = int(CellType.MOUTH)
            return BrainOutput(wants_grow=True, grow_direction=-1, grow_cell_type=ct)

        # Chase nearest edible (sensor-detected)
        edible_s, edible_d = _nearest_sector(sd, CH_EDIBLE)
        if edible_s >= 0 and edible_d < 1.0:
            return BrainOutput(move_direction=edible_s)

        # Chase nearest foreign alive
        foreign_s, foreign_d = _nearest_sector(sd, CH_FOREIGN_ALIVE)
        if foreign_s >= 0 and foreign_d < 1.0:
            return BrainOutput(move_direction=foreign_s)

        # Chase food
        food_s, food_d = _nearest_sector(sd, CH_FOOD)
        if food_s >= 0 and food_d < 1.0:
            return BrainOutput(move_direction=food_s)

        # Scripted seek direction (move toward target even without sensor contact)
        if self.seek_direction >= 0:
            return BrainOutput(move_direction=self.seek_direction)

        # Wander toward open space
        return BrainOutput(move_direction=_most_open_sector(sd))


class PreyBrain(BrainProvider):
    """Flee from danger, seek food, wander."""

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        sd = sensors.sector_data

        # Flee if danger nearby
        danger_s, danger_d = _nearest_sector(sd, CH_DANGEROUS)
        if danger_s >= 0 and danger_d < 0.3:
            flee_dir = (danger_s + 3) % 6
            if sd[flee_dir, CH_OPEN_SPACE] > 0:
                return BrainOutput(move_direction=flee_dir)
            for offset in [1, -1, 2, -2]:
                alt = (flee_dir + offset) % 6
                if sd[alt, CH_OPEN_SPACE] > 0:
                    return BrainOutput(move_direction=alt)

        # Seek food
        food_s, food_d = _nearest_sector(sd, CH_FOOD)
        if food_s >= 0 and food_d < 0.8:
            return BrainOutput(move_direction=food_s)

        # Wander
        return BrainOutput(move_direction=_most_open_sector(sd))


class WarriorBrain(BrainProvider):
    """Move toward nearest foreign organism to ram with spikes."""

    def __init__(self, seek_direction: int = -1) -> None:
        self.seek_direction = seek_direction

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        sd = sensors.sector_data

        # Chase nearest foreign (sensor-detected)
        foreign_s, foreign_d = _nearest_sector(sd, CH_FOREIGN_ALIVE)
        if foreign_s >= 0 and foreign_d < 1.0:
            return BrainOutput(move_direction=foreign_s)

        # Scripted seek
        if self.seek_direction >= 0:
            return BrainOutput(move_direction=self.seek_direction)

        # Wander
        return BrainOutput(move_direction=_most_open_sector(sd))


class ReproducerBrain(BrainProvider):
    """Grow from seed, then reproduce when energy is sufficient."""

    def __init__(self, growth_ticks: int = 60, reproduce_threshold: int = 300) -> None:
        self.growth_ticks = growth_ticks
        self.reproduce_threshold = reproduce_threshold

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        sd = sensors.sector_data

        # Growth phase
        if organism.age < self.growth_ticks:
            if organism.age < self.growth_ticks // 3:
                ct = int(CellType.PHOTOSYNTHETIC)
            elif organism.age < 2 * self.growth_ticks // 3:
                ct = int(CellType.SOFT_TISSUE)
            else:
                ct = int(CellType.SKIN)
            return BrainOutput(wants_grow=True, grow_direction=-1, grow_cell_type=ct)

        # Reproduce when enough energy — find a border cell with empty neighbor
        if organism.energy > self.reproduce_threshold and organism.cell_count >= 3:
            cell_idx, repro_dir = _find_border_reproducible_cell(organism)
            if cell_idx >= 0:
                return BrainOutput(
                    reproduce_cell_idx=cell_idx,
                    reproduce_direction=repro_dir,
                    reproduce_energy=500,
                )

        # Keep growing if energy allows
        if organism.energy > 50:
            return BrainOutput(
                wants_grow=True,
                grow_direction=-1,
                grow_cell_type=int(CellType.PHOTOSYNTHETIC),
            )

        # Seek food
        food_s, food_d = _nearest_sector(sd, CH_FOOD)
        if food_s >= 0 and food_d < 0.8 and organism.locomotion_power > 0:
            return BrainOutput(move_direction=food_s)

        return BrainOutput()

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from interfaces.brain import (
    BrainOutput, BrainProvider, OrganismView, SensorInputs,
    NUM_SECTORS, CH_FOOD, CH_FOREIGN_ALIVE, CH_DANGEROUS,
    CH_OPEN_SPACE, CH_EDIBLE,
)
from simulator.cell_types import CellType, NUM_ORGANISM_CELL_TYPES

NUM_BRAIN_PARAMS = 30

# Parameter indices
P_FLEE_THRESHOLD = 0
P_THREAT_DIST_THRESHOLD = 1
P_HUNGER_THRESHOLD = 2
P_FOOD_SEEK_RANGE = 3
P_REPRODUCE_THRESHOLD = 4
P_MIN_REPRODUCE_AGE = 5
P_GROWTH_THRESHOLD = 6
P_OFFSPRING_ENERGY = 7
# 8-13: sector movement weights for wandering (6 values)
P_WANDER_SECTOR_START = 8
# 14-21: cell type weights for growth selection (8 organism cell types)
P_CELL_TYPE_WEIGHT_START = 14
P_REPRODUCE_ENERGY_FRAC = 22
P_GROW_TOWARD_FOOD = 23
P_FLEE_SPEED = 24
P_AGGRESSION = 25
# 26-29: reserved
P_RESERVED_START = 26

ORGANISM_CELL_TYPES = [ct for ct in CellType if ct not in (CellType.NULL, CellType.FOOD)]


def default_brain_params() -> NDArray[np.float32]:
    params = np.zeros(NUM_BRAIN_PARAMS, dtype=np.float32)
    params[P_FLEE_THRESHOLD] = 0.3
    params[P_THREAT_DIST_THRESHOLD] = 0.2
    params[P_HUNGER_THRESHOLD] = 0.3
    params[P_FOOD_SEEK_RANGE] = 0.8
    params[P_REPRODUCE_THRESHOLD] = 0.5
    params[P_MIN_REPRODUCE_AGE] = 0.1
    params[P_GROWTH_THRESHOLD] = 0.15
    params[P_OFFSPRING_ENERGY] = 0.3
    params[P_WANDER_SECTOR_START:P_WANDER_SECTOR_START + 6] = 1.0 / 6.0
    params[P_CELL_TYPE_WEIGHT_START:P_CELL_TYPE_WEIGHT_START + NUM_ORGANISM_CELL_TYPES] = 1.0 / NUM_ORGANISM_CELL_TYPES
    params[P_REPRODUCE_ENERGY_FRAC] = 0.3
    params[P_GROW_TOWARD_FOOD] = 0.5
    params[P_AGGRESSION] = 0.3
    params[P_FLEE_SPEED] = 0.5
    return params


def _nearest_sector(sector_data: NDArray[np.float32], channel: int) -> tuple[int, float]:
    """Find the sector with the closest detection for a given channel."""
    best_sector = -1
    best_dist = 1.0
    for s in range(NUM_SECTORS):
        d = sector_data[s, channel]
        if d < best_dist:
            best_dist = d
            best_sector = s
    return best_sector, best_dist


def _opposite_sector(sector: int) -> int:
    return (sector + 3) % 6


def _best_open_sector(sector_data: NDArray[np.float32], weights: NDArray[np.float32]) -> int:
    scores = np.zeros(NUM_SECTORS, dtype=np.float32)
    for s in range(NUM_SECTORS):
        if sector_data[s, CH_OPEN_SPACE] > 0:
            scores[s] = weights[s]
    total = scores.sum()
    if total <= 0:
        return -1
    return int(np.argmax(scores))


class RuleBrain(BrainProvider):
    def __init__(self, params: NDArray[np.float32] | None = None) -> None:
        self.params = params if params is not None else default_brain_params()

    def evaluate(
        self,
        organism: OrganismView,
        sensors: SensorInputs,
        genome_data: dict,
    ) -> BrainOutput:
        p = self.params
        sd = sensors.sector_data
        energy_frac = sensors.own_energy
        age_frac = sensors.own_age

        danger_sector, danger_dist = _nearest_sector(sd, CH_DANGEROUS)
        food_sector, food_dist = _nearest_sector(sd, CH_FOOD)
        edible_sector, edible_dist = _nearest_sector(sd, CH_EDIBLE)

        has_locomotion = organism.locomotion_power > 0

        # Rule 1: FLEE — danger nearby
        if (has_locomotion
                and danger_sector >= 0
                and danger_dist < p[P_THREAT_DIST_THRESHOLD]):
            flee_dir = _opposite_sector(danger_sector)
            if sd[flee_dir, CH_OPEN_SPACE] > 0:
                return BrainOutput(move_direction=flee_dir)
            for offset in [1, -1, 2, -2]:
                alt = (flee_dir + offset) % 6
                if sd[alt, CH_OPEN_SPACE] > 0:
                    return BrainOutput(move_direction=alt)

        # Rule 2: ATTACK — if aggressive and enemy nearby
        if (has_locomotion
                and p[P_AGGRESSION] > 0.5
                and edible_sector >= 0
                and edible_dist < 0.3):
            return BrainOutput(move_direction=edible_sector)

        # Rule 3: SEEK FOOD — hungry
        if energy_frac < p[P_HUNGER_THRESHOLD]:
            if food_sector >= 0 and food_dist < p[P_FOOD_SEEK_RANGE]:
                if has_locomotion:
                    return BrainOutput(move_direction=food_sector)
            if edible_sector >= 0 and edible_dist < p[P_FOOD_SEEK_RANGE]:
                if has_locomotion:
                    return BrainOutput(move_direction=edible_sector)

        # Rule 4: REPRODUCE — enough energy and old enough
        if (energy_frac > p[P_REPRODUCE_THRESHOLD]
                and age_frac > p[P_MIN_REPRODUCE_AGE]
                and organism.cell_count >= 3):
            repro_energy = max(10, int(organism.energy * p[P_REPRODUCE_ENERGY_FRAC]))
            repro_cell_idx, repro_dir = self._find_reproduce_cell(organism, sd)
            if repro_cell_idx >= 0:
                return BrainOutput(
                    reproduce_cell_idx=repro_cell_idx,
                    reproduce_direction=repro_dir,
                    reproduce_energy=repro_energy,
                    wants_grow=False,
                )

        # Rule 5: GROW — have template and energy
        if energy_frac > p[P_GROWTH_THRESHOLD]:
            grow_type = self._select_grow_type()
            grow_dir = -1
            if food_sector >= 0 and p[P_GROW_TOWARD_FOOD] > 0.5:
                grow_dir = food_sector
            return BrainOutput(
                wants_grow=True,
                grow_direction=grow_dir,
                grow_cell_type=grow_type,
            )

        # Rule 6: WANDER
        if has_locomotion:
            wander_weights = p[P_WANDER_SECTOR_START:P_WANDER_SECTOR_START + 6]
            d = _best_open_sector(sd, wander_weights)
            if d >= 0:
                return BrainOutput(move_direction=d)

        return BrainOutput()

    def _select_grow_type(self) -> int:
        weights = self.params[P_CELL_TYPE_WEIGHT_START:P_CELL_TYPE_WEIGHT_START + NUM_ORGANISM_CELL_TYPES]
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if total <= 0:
            return int(CellType.SOFT_TISSUE)
        idx = int(np.argmax(weights))
        return int(ORGANISM_CELL_TYPES[idx])

    def _find_reproduce_cell(
        self,
        organism: OrganismView,
        sector_data: NDArray[np.float32],
    ) -> tuple[int, int]:
        """Find a cell with can_reproduce and a valid open direction."""
        from simulator.cell_types import CELL_PROPERTIES
        best_dir = _best_open_sector(sector_data, np.ones(NUM_SECTORS, dtype=np.float32))
        if best_dir < 0:
            best_dir = 0

        w = organism.grid_width
        for q, r, ct in organism.cells:
            props = CELL_PROPERTIES.get(CellType(ct))
            if props and props.can_reproduce:
                return (r * w + q, best_dir)
        return (-1, 0)

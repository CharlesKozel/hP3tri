"""
GPU-accelerated rule-based brain evaluation kernel.

Replaces the Python-based RuleBrain.evaluate() with a Taichi kernel
that runs in parallel across all organisms on the GPU.
"""

import taichi as ti

# Constants matching interfaces/brain.py
NUM_SECTORS: int = 6
CH_FOOD: int = 0
CH_FOREIGN_ALIVE: int = 1
CH_DEAD: int = 2
CH_DANGEROUS: int = 3
CH_FRIENDLY: int = 4
CH_OPEN_SPACE: int = 5
CH_EDIBLE: int = 6
NUM_CHANNELS: int = 7

# Brain parameter indices (matching brains/rule_brain.py)
P_FLEE_THRESHOLD: int = 0
P_THREAT_DIST_THRESHOLD: int = 1
P_HUNGER_THRESHOLD: int = 2
P_FOOD_SEEK_RANGE: int = 3
P_REPRODUCE_THRESHOLD: int = 4
P_MIN_REPRODUCE_AGE: int = 5
P_GROWTH_THRESHOLD: int = 6
P_OFFSPRING_ENERGY: int = 7
P_WANDER_SECTOR_START: int = 8
P_CELL_TYPE_WEIGHT_START: int = 14
P_REPRODUCE_ENERGY_FRAC: int = 22
P_GROW_TOWARD_FOOD: int = 23
P_FLEE_SPEED: int = 24
P_AGGRESSION: int = 25
NUM_BRAIN_PARAMS: int = 30

SENSOR_NOTHING: int = 9999


@ti.func
def _opposite_sector(sector: ti.i32) -> ti.i32:
    return (sector + 3) % 6


@ti.func
def _nearest_sector_impl(
    sensor_distances: ti.template(),
    oid: ti.i32,
    channel: ti.i32,
    max_range: ti.i32,
) -> ti.types.vector(2, ti.i32):
    """Find sector with nearest detection. Returns (sector, distance*1000)."""
    best_sector = -1
    best_dist = max_range * 1000
    for s in ti.static(range(NUM_SECTORS)):
        d = sensor_distances[oid, s, channel]
        if d > 0 and d < SENSOR_NOTHING and d < best_dist:
            best_dist = d
            best_sector = s
    return ti.Vector([best_sector, best_dist], dt=ti.i32)


@ti.func
def _best_open_sector(
    sensor_distances: ti.template(),
    oid: ti.i32,
) -> ti.i32:
    """Find best open sector (first one with open space)."""
    result = -1
    for s in ti.static(range(NUM_SECTORS)):
        if result < 0 and sensor_distances[oid, s, CH_OPEN_SPACE] > 0:
            result = s
    return result


@ti.func
def _has_open_space(
    sensor_distances: ti.template(),
    oid: ti.i32,
    sector: ti.i32,
) -> ti.i32:
    return 1 if sensor_distances[oid, sector, CH_OPEN_SPACE] > 0 else 0


@ti.func
def evaluate_brain_gpu(
    oid: ti.i32,
    organisms: ti.template(),
    sensor_distances: ti.template(),
    brain_params: ti.template(),
    ct_can_reproduce: ti.template(),
    grid: ti.template(),
    max_range: ti.i32,
    width: ti.i32,
    height: ti.i32,
):
    """
    GPU kernel function to evaluate rule-based brain for a single organism.
    Sets brain output fields directly on the organism struct.
    """
    # Initialize brain outputs to defaults
    organisms[oid].brain_move_dir = -1
    organisms[oid].brain_wants_grow = 0
    organisms[oid].brain_grow_direction = -1
    organisms[oid].brain_grow_cell_type = 1  # SOFT_TISSUE
    organisms[oid].brain_reproduce_cell_idx = -1
    organisms[oid].brain_reproduce_direction = 0
    organisms[oid].brain_reproduce_energy = 10

    # Only process if alive - wrap all logic in this check (no early returns in Taichi)
    if organisms[oid].alive == 1:
        # Get organism state
        energy = organisms[oid].energy
        age = organisms[oid].age
        cell_count = organisms[oid].cell_count
        locomotion_power = organisms[oid].locomotion_power
        genome_id = organisms[oid].genome_id

        # Normalize energy and age (assuming 1000 as typical max)
        energy_frac = ti.cast(energy, ti.f32) / 1000.0
        age_frac = ti.cast(age, ti.f32) / 1000.0

        # Load brain parameters for this genome
        p_threat_dist = brain_params[genome_id, P_THREAT_DIST_THRESHOLD]
        p_hunger = brain_params[genome_id, P_HUNGER_THRESHOLD]
        p_food_range = brain_params[genome_id, P_FOOD_SEEK_RANGE]
        p_reproduce = brain_params[genome_id, P_REPRODUCE_THRESHOLD]
        p_min_age = brain_params[genome_id, P_MIN_REPRODUCE_AGE]
        p_growth = brain_params[genome_id, P_GROWTH_THRESHOLD]
        p_aggression = brain_params[genome_id, P_AGGRESSION]
        p_grow_food = brain_params[genome_id, P_GROW_TOWARD_FOOD]
        p_repro_frac = brain_params[genome_id, P_REPRODUCE_ENERGY_FRAC]

        has_locomotion = locomotion_power > 0

        # Find nearest detections
        danger_result = _nearest_sector_impl(sensor_distances, oid, CH_DANGEROUS, max_range)
        danger_sector = danger_result[0]
        danger_dist = ti.cast(danger_result[1], ti.f32) / ti.cast(max_range * 1000, ti.f32)

        food_result = _nearest_sector_impl(sensor_distances, oid, CH_FOOD, max_range)
        food_sector = food_result[0]
        food_dist = ti.cast(food_result[1], ti.f32) / ti.cast(max_range * 1000, ti.f32)

        edible_result = _nearest_sector_impl(sensor_distances, oid, CH_EDIBLE, max_range)
        edible_sector = edible_result[0]
        edible_dist = ti.cast(edible_result[1], ti.f32) / ti.cast(max_range * 1000, ti.f32)

        done = 0

        # Rule 1: FLEE from danger
        if done == 0 and has_locomotion and danger_sector >= 0 and danger_dist < p_threat_dist:
            flee_dir = _opposite_sector(danger_sector)
            found = 0
            if _has_open_space(sensor_distances, oid, flee_dir) == 1:
                organisms[oid].brain_move_dir = flee_dir
                found = 1
            if found == 0:
                for offset in ti.static([1, -1, 2, -2]):
                    if found == 0:
                        alt = (flee_dir + offset) % 6
                        if _has_open_space(sensor_distances, oid, alt) == 1:
                            organisms[oid].brain_move_dir = alt
                            found = 1
            if found == 1:
                done = 1

        # Rule 2: ATTACK if aggressive
        if done == 0 and has_locomotion and p_aggression > 0.5 and edible_sector >= 0 and edible_dist < 0.3:
            organisms[oid].brain_move_dir = edible_sector
            done = 1

        # Rule 3: SEEK FOOD when hungry
        if done == 0 and energy_frac < p_hunger:
            if food_sector >= 0 and food_dist < p_food_range and has_locomotion:
                organisms[oid].brain_move_dir = food_sector
                done = 1
            elif done == 0 and edible_sector >= 0 and edible_dist < p_food_range and has_locomotion:
                organisms[oid].brain_move_dir = edible_sector
                done = 1

        # Rule 4: REPRODUCE when conditions met
        if done == 0 and energy_frac > p_reproduce and age_frac > p_min_age and cell_count >= 3:
            repro_energy = ti.max(10, ti.cast(ti.cast(energy, ti.f32) * p_repro_frac, ti.i32))
            repro_dir = _best_open_sector(sensor_distances, oid)
            if repro_dir < 0:
                repro_dir = 0
            organisms[oid].brain_reproduce_cell_idx = -2  # Signal to find any capable cell
            organisms[oid].brain_reproduce_direction = repro_dir
            organisms[oid].brain_reproduce_energy = repro_energy
            done = 1

        # Rule 5: GROW when energy available
        if done == 0 and energy_frac > p_growth:
            grow_dir = -1
            if food_sector >= 0 and p_grow_food > 0.5:
                grow_dir = food_sector

            # Select growth cell type based on weights (skip NULL=0 and FOOD=6)
            # Mapping: weight idx 0-7 → cell types [1,2,3,4,5,7,8,9]
            grow_type = 1  # Default SOFT_TISSUE
            max_weight = ti.cast(0.0, ti.f32)
            for ct_idx in ti.static(range(8)):
                w = brain_params[genome_id, P_CELL_TYPE_WEIGHT_START + ct_idx]
                # Map weight index to actual cell type, skipping FOOD(6)
                mapped_ct = ct_idx + 1
                if mapped_ct >= 6:
                    mapped_ct = ct_idx + 2  # Skip over FOOD
                if w > max_weight:
                    max_weight = w
                    grow_type = mapped_ct

            organisms[oid].brain_wants_grow = 1
            organisms[oid].brain_grow_direction = grow_dir
            organisms[oid].brain_grow_cell_type = grow_type
            done = 1

        # Rule 6: WANDER
        if done == 0 and has_locomotion:
            d = _best_open_sector(sensor_distances, oid)
            if d >= 0:
                organisms[oid].brain_move_dir = d


@ti.func
def find_reproduce_cell(
    oid: ti.i32,
    organisms: ti.template(),
    grid: ti.template(),
    ct_can_reproduce: ti.template(),
    grid_size: ti.i32,
    width: ti.i32,
) -> ti.i32:
    """
    Find first cell that can reproduce for this organism.
    Returns grid index or -1 if none found.
    """
    result = -1
    for idx in range(grid_size):
        if result < 0:
            if grid[idx].organism_id == oid:
                ct = ti.cast(grid[idx].cell_type, ti.i32)
                if ct_can_reproduce[ct] == 1:
                    result = idx
    return result

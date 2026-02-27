import os

import taichi as ti
import numpy as np
from numpy.typing import NDArray
from simulator.cell_types import CellTypeFields, CellType
from simulator.sim_types import OrganismId, GenomeId, DEAD
from interfaces.brain import BrainProvider, BrainOutput, OrganismView
from interfaces.body_plan import BodyPlanProvider
from interfaces.sensor import SensorAggregator
from stubs.stub_brain import StubBrain
from stubs.stub_body_plan import StubBodyPlan
from stubs.stub_sensor import StubSensor
from simulator.tick_movement import (
    compute_movers_and_priorities,
    write_claims,
    invalidate_conflicting_claims,
    copy_grid_to_temp,
    clear_mover_source_cells,
    write_mover_destination_cells,
    commit_temp_grid,
    deduct_movement_costs,
)
from simulator.tick_death import (
    mark_organism_death,
    clear_dead_cells,
    init_connectivity_labels,
    propagate_connectivity,
    count_components,
    find_best_component,
    remove_disconnected_cells,
    clear_connectivity_flags,
)
from simulator.tick_actions import (
    init_action_state,
    write_action_claims,
    execute_action_claims,
    compact_reproduce_buffer,
)
from simulator.tick_sensors import (
    init_sensor_organism,
    cell_vision,
    SENSOR_NOTHING,
)
from simulator.tick_brain import (
    evaluate_brain_gpu,
    find_reproduce_cell,
    NUM_BRAIN_PARAMS,
)
from interfaces.brain import (
    SensorInputs,
    NUM_SECTORS,
    NUM_CHANNELS,
    CH_OPEN_SPACE,
)
from simulator.cell_types import NUM_CELL_TYPES

# Maximum number of unique genomes for brain params storage
MAX_GENOMES: int = 1024

_ARCH_MAP = {"cuda": ti.cuda, "cpu": ti.cpu, "metal": ti.metal, "vulkan": ti.vulkan}
_arch = _ARCH_MAP.get(os.environ.get("TAICHI_ARCH", "cpu").lower(), ti.cpu)
ti.init(arch=_arch, default_ip=ti.i32, default_fp=ti.f32)

MAX_ORGANISMS = 2**16

@ti.dataclass
class GridSpecs:
    width: ti.u16
    height: ti.u16
    size: ti.u16

@ti.dataclass
class GridCell:
    cell_type: ti.i8
    organism_id: ti.i32
    terrain_type: ti.i8
    direction: ti.i8

# TODO check sizes of these, most are larger than necessary, or signed when unsigned would work
@ti.dataclass
class Organism:
    genome_id: ti.i32
    alive: ti.i32
    age: ti.i32

    energy: ti.i32
    cell_count: ti.i32
    total_mass: ti.i32
    movement_points: ti.i32

    locomotion_power: ti.i32
    upkeep_cost: ti.i32
    energy_generation: ti.i32

    movement_priority: ti.u64
    can_move: ti.i8
    brain_move_dir: ti.i32
    brain_wants_grow: ti.i32
    brain_grow_direction: ti.i32
    brain_grow_cell_type: ti.i32
    brain_reproduce_cell_idx: ti.i32
    brain_reproduce_direction: ti.i32
    brain_reproduce_energy: ti.i32
    brain_reproduce_cell_type: ti.i32
    growth_cells_placed: ti.i32
    reproduce_cells_placed: ti.i32
    needs_connectivity_check: ti.i8
    cell_type_counts: ti.types.vector(NUM_CELL_TYPES, ti.i32)


@ti.data_oriented
class SimulationEngine:
    def __init__(self, width: int, height: int, seed: int = 42) -> None:
        self.width = width
        self.height = height
        self.grid_size = width * height
        self.tick_count = 0
        self.grid_specs = GridSpecs(width, height, self.grid_size)

        self.next_org_id: ti.u16 = 1 # start at 1, 0 = no organism
        self.organism_genome_map: dict[OrganismId, GenomeId] = {}
        self.organism_brain_map: dict[OrganismId, BrainProvider] = {}
        self.organism_body_plan_map: dict[OrganismId, BodyPlanProvider] = {}
        self.genome_registry: dict[GenomeId, dict] = {}
        self.rng = np.random.default_rng(seed)

        self.default_brain: BrainProvider = StubBrain()
        self.default_body_plan: BodyPlanProvider = StubBodyPlan()
        self.default_sensor: SensorAggregator = StubSensor()

        # Taichi Fields
        self.grid = GridCell.field(shape=self.grid_size)
        self.temp_grid = GridCell.field(shape=self.grid_size)
        self.organisms = Organism.field(shape=MAX_ORGANISMS)
        self.claims = ti.field(dtype=ti.u64, shape=self.grid_size)

        # Scratch fields
        self.org_move_direction = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_move_priority = ti.field(dtype=ti.i64, shape=MAX_ORGANISMS)
        self.org_move_success = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)

        self.claim_combined = ti.field(dtype=ti.i64, shape=self.grid_size)
        self.temp_cell_type = ti.field(dtype=ti.i8, shape=self.grid_size)
        self.temp_organism_id = ti.field(dtype=ti.i32, shape=self.grid_size)

        self.action_claims = ti.field(dtype=ti.i64, shape=self.grid_size)
        self.reproduce_buffer = ti.field(dtype=ti.i32, shape=self.grid_size)

        # Sparse reproduce buffer for efficient GPU->CPU transfer
        # Only transfer actual reproductions (typically few per tick) instead of entire grid
        max_reproduce_per_tick = min(MAX_ORGANISMS // 2, 4096)
        self.reproduce_idx_buffer = ti.field(dtype=ti.i32, shape=max_reproduce_per_tick)
        self.reproduce_oid_buffer = ti.field(dtype=ti.i32, shape=max_reproduce_per_tick)
        self.reproduce_count = ti.field(dtype=ti.i32, shape=())
        self.max_reproduce_per_tick = max_reproduce_per_tick

        # Connectivity fields
        self.labels = ti.field(dtype=ti.i32, shape=self.grid_size)
        self.connectivity_changed = ti.field(dtype=ti.i32, shape=())
        self.any_connectivity_needed = ti.field(dtype=ti.i32, shape=())
        self.component_size = ti.field(dtype=ti.i32, shape=self.grid_size)
        self.best_component = ti.field(dtype=ti.i64, shape=MAX_ORGANISMS)

        # Sensor fields
        self.sensor_distances = ti.field(dtype=ti.i32, shape=(MAX_ORGANISMS, NUM_SECTORS, NUM_CHANNELS))

        # Brain parameters field (per-genome)
        # Stores rule-based brain params on GPU for all genomes
        self.brain_params = ti.field(dtype=ti.f32, shape=(MAX_GENOMES, NUM_BRAIN_PARAMS))
        self._init_default_brain_params()

        # GPU brain mode flag
        self.use_gpu_brain = True

        # Cell type property lookup tables
        self.ct_fields = CellTypeFields()
        self.ct_fields.load()

    def _init_default_brain_params(self) -> None:
        """Initialize default brain parameters for genome 0."""
        defaults = np.zeros(NUM_BRAIN_PARAMS, dtype=np.float32)
        defaults[0] = 0.3   # P_FLEE_THRESHOLD
        defaults[1] = 0.2   # P_THREAT_DIST_THRESHOLD
        defaults[2] = 0.3   # P_HUNGER_THRESHOLD
        defaults[3] = 0.8   # P_FOOD_SEEK_RANGE
        defaults[4] = 0.5   # P_REPRODUCE_THRESHOLD
        defaults[5] = 0.1   # P_MIN_REPRODUCE_AGE
        defaults[6] = 0.15  # P_GROWTH_THRESHOLD
        defaults[7] = 0.3   # P_OFFSPRING_ENERGY
        defaults[8:14] = 1.0 / 6.0   # P_WANDER_SECTOR weights
        defaults[14:22] = 1.0 / 8.0  # P_CELL_TYPE weights
        defaults[22] = 0.3  # P_REPRODUCE_ENERGY_FRAC
        defaults[23] = 0.5  # P_GROW_TOWARD_FOOD
        defaults[24] = 0.5  # P_FLEE_SPEED
        defaults[25] = 0.3  # P_AGGRESSION
        # Set for all genomes as default
        for g in range(MAX_GENOMES):
            for i in range(NUM_BRAIN_PARAMS):
                self.brain_params[g, i] = float(defaults[i])

    def set_genome_brain_params(self, genome_id: int, params: np.ndarray) -> None:
        """Set brain parameters for a specific genome."""
        for i in range(min(len(params), NUM_BRAIN_PARAMS)):
            self.brain_params[genome_id, i] = float(params[i])

    # ── Full tick ───────────────────────────────────────────────
    def step(self) -> None:
        # Tick 0 is initial conditions, no actions / brain eval
        self.tick_count += 1

        self.recompute_aggregates()

        self.apply_resources()

        self.step_sensors()

        self.step_brains()

        self.process_movement()

        self.step_actions()

        self.process_death_and_disconnection()

        self.increment_ages()

    def create_organism(
        self,
        seed_q: int,
        seed_r: int,
        seed_cell_type: int,
        starting_energy: int,
        genome_id: GenomeId = 0,
        brain: BrainProvider | None = None,
        body_plan: BodyPlanProvider | None = None,
        seed_direction: int = 0,
    ) -> OrganismId:
        org_id = self.next_org_id
        self.next_org_id += 1

        self.organisms[org_id].alive = 1
        self.organisms[org_id].genome_id = genome_id
        self.organisms[org_id].energy = starting_energy
        self.organisms[org_id].age = 0
        self.organisms[org_id].movement_points = 0

        grid_idx = seed_r * self.width + seed_q
        self.grid[grid_idx].cell_type = seed_cell_type
        self.grid[grid_idx].organism_id = org_id
        self.grid[grid_idx].direction = seed_direction

        self.organism_genome_map[org_id] = genome_id
        if brain is not None:
            self.organism_brain_map[org_id] = brain
        if body_plan is not None:
            self.organism_body_plan_map[org_id] = body_plan

        return org_id

    def place_cell(self, org_id: OrganismId, q: int, r: int, cell_type: int, direction: int = 0) -> None:
        grid_idx = r * self.width + q
        self.grid[grid_idx].cell_type = cell_type
        self.grid[grid_idx].organism_id = org_id
        self.grid[grid_idx].direction = direction

    # ── Step 0: Recompute aggregates ────────────────────────────

    def recompute_aggregates(self) -> None:
        # Clear and recompute in a single fused kernel for efficiency
        self._kernel_recompute_aggregates(
            self.grid,
            self.ct_fields.mass,
            self.ct_fields.maintenance_cost,
            self.ct_fields.locomotion_power,
            self.ct_fields.energy_generation,
            self.organisms,
            self.next_org_id,
            self.grid_size,
        )

    @ti.kernel
    def _kernel_recompute_aggregates(
        self,
        grid: ti.template(),
        ct_mass: ti.template(),
        ct_maintenance: ti.template(),
        ct_locomotion: ti.template(),
        ct_energy_gen: ti.template(),
        org: ti.template(),
        next_org_id: ti.i32,
        grid_size: ti.i32,
    ):
        # First clear aggregates (fused for efficiency - avoids fill() overhead)
        for oid in range(next_org_id):
            org[oid].cell_count = 0
            org[oid].total_mass = 0
            org[oid].locomotion_power = 0
            org[oid].upkeep_cost = 0
            org[oid].energy_generation = 0
            for ct_idx in ti.static(range(NUM_CELL_TYPES)):
                org[oid].cell_type_counts[ct_idx] = 0
        # Then accumulate from grid
        for idx in range(grid_size):
            oid = grid[idx].organism_id
            if oid > 0:
                ct = ti.cast(grid[idx].cell_type, ti.i32)
                ti.atomic_add(org[oid].cell_count, 1)
                ti.atomic_add(org[oid].total_mass, ct_mass[ct])
                ti.atomic_add(org[oid].locomotion_power, ct_locomotion[ct])
                ti.atomic_add(org[oid].upkeep_cost, ct_maintenance[ct])
                ti.atomic_add(org[oid].energy_generation, ct_energy_gen[ct])
                ti.atomic_add(org[oid].cell_type_counts[ct], 1)

    def apply_resources(self) -> None:
        self._kernel_apply_resources(self.organisms, self.next_org_id)

    @ti.kernel
    def _kernel_apply_resources(
        self,
        org: ti.template(),
        next_org_id: ti.i32,
    ):
        for oid in range(next_org_id):
            if org[oid].alive == 1:
                org[oid].energy += org[oid].energy_generation
                org[oid].energy -= org[oid].upkeep_cost

    # ── Step 2: Sensor aggregation ──────────────────────────────

    def step_sensors(self) -> None:
        self._sensor_init(
            self.sensor_distances,
            self.next_org_id,
        )
        self._sensor_scan(
            self.grid, self.organisms,
            self.ct_fields.vision_range,
            self.ct_fields.vision_expansion,
            self.ct_fields.directional,
            self.ct_fields.can_eat,
            self.ct_fields.can_destroy,
            self.sensor_distances,
            self.width, self.height, self.grid_size,
        )

    @ti.kernel
    def _sensor_init(
        self,
        sensor_distances: ti.template(),
        next_org_id: ti.i32,
    ):
        for oid in range(next_org_id):
            init_sensor_organism(oid, sensor_distances)

    @ti.kernel
    def _sensor_scan(
        self,
        grid: ti.template(),
        organisms: ti.template(),
        ct_vision_range: ti.template(),
        ct_vision_expansion: ti.template(),
        ct_directional: ti.template(),
        can_eat: ti.template(),
        can_destroy: ti.template(),
        sensor_distances: ti.template(),
        width: ti.i32,
        height: ti.i32,
        grid_size: ti.i32,
    ):
        for idx in range(grid_size):
            oid = grid[idx].organism_id
            if oid > 0 and organisms[oid].alive == 1:
                cell_vision(idx, grid, organisms,
                            ct_vision_range, ct_vision_expansion, ct_directional,
                            can_eat, can_destroy, sensor_distances, width, height)

    def build_sensor_inputs(self, org_id: int) -> SensorInputs:
        raw = np.zeros((NUM_SECTORS, NUM_CHANNELS), dtype=np.float32)
        max_range = max(self.width, self.height) // 2
        for s in range(NUM_SECTORS):
            for c in range(NUM_CHANNELS):
                val = int(self.sensor_distances[org_id, s, c])
                if c == CH_OPEN_SPACE:
                    raw[s, c] = float(min(val, 1))
                elif val >= SENSOR_NOTHING:
                    raw[s, c] = 1.0
                else:
                    raw[s, c] = val / max_range if max_range > 0 else 1.0
        cell_counts = np.array(
            [int(self.organisms[org_id].cell_type_counts[ct]) for ct in range(NUM_CELL_TYPES)],
            dtype=np.int32,
        )
        energy = int(self.organisms[org_id].energy)
        age = int(self.organisms[org_id].age)
        return SensorInputs(
            sector_data=raw,
            own_energy=energy / 1000.0,
            own_age=age / 1000.0,
            own_cell_counts=cell_counts,
        )

    # ── Step 3: Brain evaluation ────────────────────────────────

    def _build_organism_view(
        self, org_id: int,
        ct_np: "NDArray[np.int8]",
        oid_np: "NDArray[np.int32]",
    ) -> OrganismView:
        cells: list[tuple[int, int, int]] = []
        for idx in range(self.grid_size):
            if int(oid_np[idx]) == org_id:
                q = idx % self.width
                r = idx // self.width
                cells.append((q, r, int(ct_np[idx])))
        return OrganismView(
            organism_id=org_id,
            age=int(self.organisms[org_id].age),
            energy=int(self.organisms[org_id].energy),
            cell_count=int(self.organisms[org_id].cell_count),
            total_mass=int(self.organisms[org_id].total_mass),
            locomotion_power=int(self.organisms[org_id].locomotion_power),
            cells=cells,
            grid_width=self.width,
            grid_height=self.height,
        )

    def step_brains(self) -> None:
        """Evaluate brains for all organisms - GPU or CPU path."""
        if self.use_gpu_brain:
            self._step_brains_gpu()
        else:
            self._step_brains_cpu()

    def _step_brains_gpu(self) -> None:
        """GPU-accelerated brain evaluation using Taichi kernel."""
        max_range = max(self.width, self.height) // 2
        self._kernel_brain_eval(
            self.organisms,
            self.sensor_distances,
            self.brain_params,
            self.ct_fields.can_reproduce,
            self.grid,
            max_range,
            self.width,
            self.height,
            self.grid_size,
            self.next_org_id,
        )
        # Handle reproduce cell finding for organisms that requested it
        self._kernel_find_reproduce_cells(
            self.organisms,
            self.grid,
            self.ct_fields.can_reproduce,
            self.grid_size,
            self.width,
            self.next_org_id,
        )

    @ti.kernel
    def _kernel_brain_eval(
        self,
        organisms: ti.template(),
        sensor_distances: ti.template(),
        brain_params: ti.template(),
        ct_can_reproduce: ti.template(),
        grid: ti.template(),
        max_range: ti.i32,
        width: ti.i32,
        height: ti.i32,
        grid_size: ti.i32,
        next_org_id: ti.i32,
    ):
        for oid in range(next_org_id):
            evaluate_brain_gpu(
                oid, organisms, sensor_distances, brain_params,
                ct_can_reproduce, grid, max_range, width, height
            )

    @ti.kernel
    def _kernel_find_reproduce_cells(
        self,
        organisms: ti.template(),
        grid: ti.template(),
        ct_can_reproduce: ti.template(),
        grid_size: ti.i32,
        width: ti.i32,
        next_org_id: ti.i32,
    ):
        """Find reproduce cells for organisms that requested reproduction."""
        for oid in range(next_org_id):
            if organisms[oid].brain_reproduce_cell_idx == -2:  # Signal to find
                result = find_reproduce_cell(oid, organisms, grid, ct_can_reproduce, grid_size, width)
                organisms[oid].brain_reproduce_cell_idx = result

    def _step_brains_cpu(self) -> None:
        """Original CPU-based brain evaluation (fallback)."""
        ct_np = self.grid.cell_type.to_numpy()
        oid_np = self.grid.organism_id.to_numpy()
        for org_id in range(1, self.next_org_id):
            if self.organisms[org_id].alive == DEAD:
                continue

            brain = self.organism_brain_map.get(org_id, self.default_brain)
            sensors = self.build_sensor_inputs(org_id)
            view = self._build_organism_view(org_id, ct_np, oid_np)
            genome_id = self.organism_genome_map.get(org_id, 0)
            genome_data = self.genome_registry.get(genome_id, {})
            outputs = brain.evaluate(view, sensors, genome_data)

            self.organisms[org_id].brain_move_dir = outputs.move_direction
            self.organisms[org_id].brain_wants_grow = 1 if outputs.wants_grow else 0
            self.organisms[org_id].brain_grow_direction = outputs.grow_direction
            self.organisms[org_id].brain_grow_cell_type = outputs.grow_cell_type
            self.organisms[org_id].brain_reproduce_cell_idx = outputs.reproduce_cell_idx
            self.organisms[org_id].brain_reproduce_direction = outputs.reproduce_direction
            self.organisms[org_id].brain_reproduce_energy = outputs.reproduce_energy


    # ── Step 4: Movement kernels ──────────────────────────────────

    def process_movement(self):
        # Note: claims.fill(0) moved into kernel for efficiency
        self.step_movement(
            self.grid,
            self.temp_grid,
            self.organisms,
            self.next_org_id,
            self.width,
            self.height,
            self.grid_size,
            self.claims,
        )

    @ti.kernel
    def step_movement(
            self,
            grid: ti.template(),
            temp_grid: ti.template(),
            organisms: ti.template(),
            next_org_id: ti.i32,
            width: ti.i32,
            height: ti.i32,
            grid_size: ti.i32,
            claims: ti.template(),
    ):
        # Clear claims in kernel (avoids fill() overhead)
        for idx in range(grid_size):
            claims[idx] = 0
        for oid in range(next_org_id):
            compute_movers_and_priorities(oid, organisms)
        for idx in range(grid_size):
            write_claims(idx, organisms, grid, width, height, grid_size, claims)
        for idx in range(grid_size):
            invalidate_conflicting_claims(idx, organisms, grid, width, height, grid_size, claims)
        for idx in range(grid_size):
            copy_grid_to_temp(idx, grid, temp_grid)
        for idx in range(grid_size):
            clear_mover_source_cells(idx, grid, organisms, temp_grid)
        for idx in range(grid_size):
            write_mover_destination_cells(idx, grid, organisms, width, height, grid_size, temp_grid)
        for idx in range(grid_size):
            commit_temp_grid(idx, grid, temp_grid)
        for oid in range(next_org_id):
            deduct_movement_costs(oid, organisms)

    # ── Step 5+6: Unified Action Resolution (eat + growth) ─────

    def step_actions(self) -> None:
        # Reset reproduce counter
        self.reproduce_count[None] = 0

        # Note: fill operations moved into kernel for efficiency
        self._kernel_step_actions(
            self.grid,
            self.organisms,
            self.ct_fields.action_rank,
            self.ct_fields.can_eat,
            self.ct_fields.can_destroy,
            self.ct_fields.can_reproduce,
            self.ct_fields.consumption_value,
            self.ct_fields.growth_cost,
            self.action_claims,
            self.reproduce_buffer,
            self.reproduce_idx_buffer,
            self.reproduce_oid_buffer,
            self.reproduce_count,
            self.max_reproduce_per_tick,
            self.next_org_id,
            self.width,
            self.height,
        )
        self._process_reproduce_buffer()

    @ti.kernel
    def _kernel_step_actions(
        self,
        grid: ti.template(),
        organisms: ti.template(),
        ct_action_rank: ti.template(),
        ct_can_eat: ti.template(),
        ct_can_destroy: ti.template(),
        ct_can_reproduce: ti.template(),
        ct_consumption_value: ti.template(),
        ct_growth_cost: ti.template(),
        action_claims: ti.template(),
        reproduce_buffer: ti.template(),
        reproduce_idx_buffer: ti.template(),
        reproduce_oid_buffer: ti.template(),
        reproduce_count: ti.template(),
        max_reproduce: ti.i32,
        next_org_id: ti.i32,
        width: ti.i32,
        height: ti.i32,
    ):
        grid_size = width * height
        # Clear buffers and init action state in first pass (fused for efficiency)
        for idx in range(grid_size):
            action_claims[idx] = 0
            reproduce_buffer[idx] = 0
        for oid in range(next_org_id):
            init_action_state(oid, organisms)
        for idx in range(grid_size):
            write_action_claims(idx, grid, organisms, ct_action_rank, ct_can_eat, ct_can_destroy, ct_can_reproduce, ct_growth_cost, action_claims, width, height)
        for idx in range(grid_size):
            execute_action_claims(idx, grid, organisms, ct_consumption_value, ct_growth_cost, action_claims, reproduce_buffer)
        # Compact sparse reproduce buffer for efficient GPU->CPU transfer
        for idx in range(grid_size):
            compact_reproduce_buffer(idx, reproduce_buffer, reproduce_idx_buffer, reproduce_oid_buffer, reproduce_count, max_reproduce)

    def _process_reproduce_buffer(self) -> None:
        # Use sparse buffer for efficient transfer (only transfer actual reproductions)
        count = int(self.reproduce_count[None])
        if count == 0:
            return  # Early exit if no reproductions

        # Only transfer the small sparse buffers (up to count entries)
        idx_buf = self.reproduce_idx_buffer.to_numpy()[:count]
        oid_buf = self.reproduce_oid_buffer.to_numpy()[:count]

        for i in range(count):
            idx = int(idx_buf[i])
            parent_oid = int(oid_buf[i])
            q = int(idx % self.width)
            r = int(idx // self.width)
            parent_genome = self.organism_genome_map[parent_oid]
            seed_ct = int(self.organisms[parent_oid].brain_reproduce_cell_type)
            energy = int(self.organisms[parent_oid].brain_reproduce_energy)
            brain = self.organism_brain_map.get(parent_oid)
            body_plan = self.organism_body_plan_map.get(parent_oid)
            self.create_organism(
                q, r, seed_ct, energy,
                genome_id=parent_genome,
                brain=brain,
                body_plan=body_plan,
            )

    # ── Step 7: Death and cleanup ───────────────────────────────

    def process_death_and_disconnection(self):
        self._kernel_step_death(
            self.grid,
            self.organisms,
            self.next_org_id,
            self.grid_size,
        )

        # Check if ANY organism needs connectivity check (early exit optimization)
        self.any_connectivity_needed[None] = 0
        self._check_any_connectivity_needed(self.organisms, self.next_org_id, self.any_connectivity_needed)
        if self.any_connectivity_needed[None] == 0:
            return  # No connectivity work needed - huge speedup!

        self._connectivity_init(
            self.grid, self.organisms, self.labels, self.grid_size,
        )
        while True:
            self.connectivity_changed[None] = 0
            self._connectivity_propagate(
                self.grid, self.organisms, self.labels,
                self.connectivity_changed, self.width, self.height, self.grid_size,
            )
            if self.connectivity_changed[None] == 0:
                break
        self.component_size.fill(0)
        self.best_component.fill(0)
        self._connectivity_resolve(
            self.grid, self.organisms, self.labels,
            self.component_size, self.best_component,
            self.next_org_id, self.grid_size,
        )

    @ti.kernel
    def _check_any_connectivity_needed(
        self,
        organisms: ti.template(),
        next_org_id: ti.i32,
        any_needed: ti.template(),
    ):
        for oid in range(next_org_id):
            if organisms[oid].needs_connectivity_check == 1:
                any_needed[None] = 1

    @ti.kernel
    def _kernel_step_death(
        self,
        grid: ti.template(),
        organisms: ti.template(),
        next_org_id: ti.i32,
        grid_size: ti.i32,
    ):
        for oid in range(next_org_id):
            mark_organism_death(oid, organisms)
        for idx in range(grid_size):
            clear_dead_cells(idx, grid, organisms)
        # NOTE: needs_connectivity_check is already set by tick_actions when cells are destroyed
        # Don't blindly set it for ALL organisms - that's wasteful!

    @ti.kernel
    def _connectivity_init(
        self,
        grid: ti.template(),
        organisms: ti.template(),
        labels: ti.template(),
        grid_size: ti.i32,
    ):
        for idx in range(grid_size):
            init_connectivity_labels(idx, grid, organisms, labels)

    @ti.kernel
    def _connectivity_propagate(
        self,
        grid: ti.template(),
        organisms: ti.template(),
        labels: ti.template(),
        changed: ti.template(),
        width: ti.i32,
        height: ti.i32,
        grid_size: ti.i32,
    ):
        for idx in range(grid_size):
            propagate_connectivity(idx, grid, organisms, labels, changed, width, height)

    @ti.kernel
    def _connectivity_resolve(
        self,
        grid: ti.template(),
        organisms: ti.template(),
        labels: ti.template(),
        component_size: ti.template(),
        best_component: ti.template(),
        next_org_id: ti.i32,
        grid_size: ti.i32,
    ):
        for idx in range(grid_size):
            count_components(idx, grid, organisms, labels, component_size)
        for idx in range(grid_size):
            find_best_component(idx, grid, organisms, labels, component_size, best_component)
        for idx in range(grid_size):
            remove_disconnected_cells(idx, grid, organisms, labels, best_component)
        for oid in range(next_org_id):
            clear_connectivity_flags(oid, organisms)

    @ti.kernel
    def _kernel_increment_ages(
        self,
        org: ti.template(),
        count: ti.i32,
    ):
        for i in range(count):
            if org[i].alive == 1:
                org[i].age += 1

    def increment_ages(self) -> None:
        self._kernel_increment_ages(
            self.organisms,
            self.next_org_id,
        )


    # ── Snapshot ────────────────────────────────────────────────

    def snapshot(self) -> dict:
        ct: NDArray[np.int8] = self.grid.cell_type.to_numpy()
        oi: NDArray[np.int32] = self.grid.organism_id.to_numpy()
        tt: NDArray[np.int8] = self.grid.terrain_type.to_numpy()

        tiles: list[dict] = []
        for idx in range(self.grid_size):
            cell = int(ct[idx])
            terrain = int(tt[idx])
            if cell != CellType.NULL:
                q = idx % self.width
                r = idx // self.width
                tiles.append({
                    "q": q,
                    "r": r,
                    "terrainType": terrain,
                    "cellType": cell,
                    "organismId": int(oi[idx]),
                })

        organisms: list[dict] = []
        for oid in range(1, self.next_org_id):
            organisms.append({
                "id": oid,
                "genomeId": int(self.organisms[oid].genome_id),
                "energy": int(self.organisms[oid].energy),
                "alive": bool(self.organisms[oid].alive),
                "cellCount": int(self.organisms[oid].cell_count),
            })

        any_alive = any(
            self.organisms[oid].alive for oid in range(1, self.next_org_id)
        )

        return {
            "tick": self.tick_count,
            "status": "RUNNING" if any_alive else "FINISHED",
            "grid": {
                "width": self.width,
                "height": self.height,
                "tiles": tiles,
            },
            "organisms": organisms,
        }
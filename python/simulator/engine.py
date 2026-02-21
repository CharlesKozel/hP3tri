import taichi as ti
import numpy as np
from numpy.typing import NDArray
from simulator.hex_grid import TerrainType, NEIGHBOR_OFFSETS, neighbor_offset
from simulator.cell_types import CellTypeFields, CellType
from simulator.sim_types import OrganismId, GenomeId, GenomeData, DEAD
from interfaces.brain import BrainProvider, OrganismView, SensorInputs, BrainOutput
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

ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)

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

    movement_priority: ti.u64
    can_move: ti.i8
    brain_move_dir: ti.i32
    brain_wants_grow: ti.i32
    brain_wants_reproduce: ti.i32


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
        self.rng = np.random.default_rng(seed)

        self.default_brain: BrainProvider = StubBrain()
        self.default_body_plan: BodyPlanProvider = StubBodyPlan()
        self.default_sensor: SensorAggregator = StubSensor()

        self._organisms_took_damage: set[OrganismId] = set()

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

        self.grow_claim_combined = ti.field(dtype=ti.i64, shape=self.grid_size)

        # Cell type property lookup tables
        self.ct_fields = CellTypeFields()
        self.ct_fields.load()

    # ── Full tick ───────────────────────────────────────────────
    def step(self) -> None:
        self.tick_count += 1

        self.recompute_aggregates()
        # self.compute_zero_cell_death()
        # self.step_resources() # TODO implement photosynthesis etc.
        # sensor_map = self.step_sensors() # removing for minimal sim test
        self.step_brains() # TODO read sensors for brains

        self.claims.fill(0)
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

        # self.step_growth() # removing for minimal sim test
        # self._organisms_took_damage.clear()
        # self.step_actions() # removing for minimal sim test
        # self.step_death()
        self._increment_ages()

    def create_organism(
        self,
        seed_q: int,
        seed_r: int,
        seed_cell_type: int,
        starting_energy: int,
        genome_id: GenomeId = 0,
        brain: BrainProvider | None = None,
        body_plan: BodyPlanProvider | None = None,
    ) -> OrganismId:
        # org_id = ti.atomic_add(self.next_org_id, 1)
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

        self.organism_genome_map[org_id] = genome_id
        if brain is not None:
            self.organism_brain_map[org_id] = brain
        if body_plan is not None:
            self.organism_body_plan_map[org_id] = body_plan

        return org_id

    # def place_cell(self, org_id: OrganismId, q: int, r: int, cell_type: int) -> None:
    #     grid_idx = r * self.width + q
    #     self.grid[grid_idx].cell_type = cell_type
    #     self.grid[grid_idx].organism_id = org_id
    #
    # def get_brain(self, org_id: OrganismId) -> BrainProvider:
    #     return self.organism_brain_map.get(org_id, self.default_brain)
    #
    # def get_body_plan(self, org_id: OrganismId) -> BodyPlanProvider:
    #     return self.organism_body_plan_map.get(org_id, self.default_body_plan)

    # ── Step 0: Recompute aggregates ────────────────────────────

    def recompute_aggregates(self) -> None:
        self.organisms.cell_count.fill(0)
        self.organisms.total_mass.fill(0)
        self.organisms.locomotion_power.fill(0)
        self.organisms.upkeep_cost.fill(0)

        self._kernel_recompute_aggregates(
            self.grid,
            self.ct_fields.mass,
            self.ct_fields.maintenance_cost,
            self.ct_fields.locomotion_power,
            self.ct_fields.energy_generation,
            self.organisms,
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
        grid_size: ti.i32,
    ):
        for idx in range(grid_size):
            oid = grid[idx].organism_id
            if oid > 0:
                ct = ti.cast(grid[idx].cell_type, ti.i32)
                arr_idx = oid - 1
                ti.atomic_add(org[arr_idx].cell_count, 1)
                ti.atomic_add(org[arr_idx].total_mass, ct_mass[ct])
                ti.atomic_add(org[arr_idx].locomotion_power, ct_locomotion[ct])
                ti.atomic_add(org[arr_idx].upkeep_cost, ct_maintenance[ct])

    # @ti.kernel
    # def _kernel_apply_resources(
    #     self,
    #     org: ti.template(),
    #     org_count: ti.i32,
    # ):
    #     for i in range(org_count):
    #         if org[i].alive == 1:
    #             org[i].energy += org[i].photosynthesis
    #             org[i].energy -= org[i].upkeep_cost


    # ── Step 2: Sensor aggregation ──────────────────────────────

    # def step_sensors(self) -> dict[OrganismId, SensorInputs]:
    #     grid_ct = self.grid.cell_type.to_numpy()
    #     grid_oid = self.grid.organism_id.to_numpy()
    #
    #     results: dict[OrganismId, SensorInputs] = {}
    #     for org_idx in range(self.next_org_id):
    #         if self.organisms[org_idx].alive != 1:
    #             continue
    #         org_id = org_idx + 1
    #         view = self._build_organism_view(org_idx, grid_ct, grid_oid)
    #         sensor = self.default_sensor
    #         results[org_id] = sensor.aggregate(
    #             view, grid_ct, grid_oid, self.width, self.height,
    #         )
    #     return results

    # ── Step 3: Brain evaluation ────────────────────────────────

    # def step_brains(self, sensor_map: dict[OrganismId, SensorInputs]) -> None:
    def step_brains(self) -> None:
        for org_id in range(self.next_org_id):
            if self.organisms[org_id].alive == DEAD:
                continue

            # view = self._build_organism_view(org_id, grid_ct, grid_oid)
            # sensors = sensor_map.get(org_id, SensorInputs())
            # outputs = brain_tick(self.organisms[org_id])

            age = self.organisms[org_id].age
            loop = int((-1 + (1 + 4 * age / 3) ** 0.5) / 2) + 1
            loop_start = 3 * (loop - 1) * loop
            pos_in_loop = age - loop_start
            direction = (pos_in_loop // loop + org_id % 6) % 6

            outputs = BrainOutput(
                move_direction=direction,
            )

            self.organisms[org_id].brain_move_dir = outputs.move_direction


    # ── Step 4: Movement kernels ──────────────────────────────────

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

    # ── Step 5: Growth ──────────────────────────────────────────

    # def step_growth(self) -> None:
    #     from simulator.tick_growth import execute_growth
    #     execute_growth(self)

    # ── Step 6: Action resolution ───────────────────────────────
    #
    # @ti.kernel
    # def _kernel_resolve_actions(
    #     self,
    #     grid: ti.template(),
    #     org: ti.template(),
    #     ct_consumption_value: ti.template(),
    #     width: ti.i32,
    #     height: ti.i32,
    #     grid_size: ti.i32,
    # ):
    #     MOUTH = ti.cast(CellType.MOUTH, ti.i32)
    #     TEETH = ti.cast(CellType.TEETH, ti.i32)
    #     SPIKE = ti.cast(CellType.SPIKE, ti.i32)
    #
    #     for idx in range(grid_size):
    #         ct = ti.cast(grid[idx].cell_type, ti.i32)
    #         oid = grid[idx].organism_id
    #         if oid <= 0:
    #             continue
    #         arr_idx = oid - 1
    #         if org[arr_idx].alive == 0:
    #             continue
    #
    #         q = idx % width
    #         r = idx // width
    #
    #         if ct == MOUTH or ct == TEETH:
    #             ate = 0
    #             for d in ti.static(range(6)):
    #                 if ate == 0:
    #                     nq = (q + NEIGHBOR_OFFSETS[d][0]) % width
    #                     nr = (r + NEIGHBOR_OFFSETS[d][1]) % height
    #                     nidx = nr * width + nq
    #                     neighbor_ct = ti.cast(grid[nidx].cell_type, ti.i32)
    #                     neighbor_oid = grid[nidx].organism_id
    #
    #                     if neighbor_ct != 0:
    #                         can_eat = False
    #                         if neighbor_oid == 0:
    #                             can_eat = True
    #                         elif neighbor_oid != oid:
    #                             if org[neighbor_oid - 1].genome_id != org[arr_idx].genome_id:
    #                                 can_eat = True
    #
    #                         if can_eat:
    #                             energy_gained = ct_consumption_value[neighbor_ct]
    #                             ti.atomic_add(org[arr_idx].energy, energy_gained)
    #                             grid[nidx].cell_type = ti.cast(0, ti.i8)
    #                             grid[nidx].organism_id = 0
    #                             ate = 1
    #
    #         if ct == SPIKE:
    #             for d in ti.static(range(6)):
    #                 nq = (q + NEIGHBOR_OFFSETS[d][0]) % width
    #                 nr = (r + NEIGHBOR_OFFSETS[d][1]) % height
    #                 nidx = nr * width + nq
    #                 neighbor_oid = grid[nidx].organism_id
    #                 neighbor_ct = ti.cast(grid[nidx].cell_type, ti.i32)
    #
    #                 if neighbor_ct != 0 and neighbor_oid > 0 and neighbor_oid != oid:
    #                     if org[neighbor_oid - 1].genome_id != org[arr_idx].genome_id:
    #                         grid[nidx].cell_type = ti.cast(0, ti.i8)
    #                         grid[nidx].organism_id = 0
    #
    # def step_actions(self) -> None:
    #     from simulator.tick_actions import execute_actions
    #     execute_actions(self)

    # ── Step 7: Death and cleanup ───────────────────────────────

    # @ti.kernel
    # def _kernel_mark_dead_cells(
    #     self,
    #     grid: ti.template(),
    #     org: ti.template(),
    #     grid_size: ti.i32,
    # ):
    #     for idx in range(grid_size):
    #         oid = grid[idx].organism_id
    #         if oid > 0 and org[oid - 1].alive == 0:
    #             grid[idx].organism_id = 0
    #
    # def step_death(self) -> None:
    #     from simulator.tick_death import execute_death
    #     execute_death(self)
    #
    # def compute_zero_cell_death(self) -> None:
    #     for org_idx in range(self.next_org_id):
    #         if self.organisms[org_idx].alive == 1 and self.organisms[org_idx].cell_count == 0:
    #             self.organisms[org_idx].alive = 0

    @ti.kernel
    def _kernel_increment_ages(
        self,
        org: ti.template(),
        count: ti.i32,
    ):
        for i in range(count):
            if org[i].alive == 1:
                org[i].age += 1

    def _increment_ages(self) -> None:
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
        for i in range(self.next_org_id):
            organisms.append({
                "id": i + 1,
                "energy": int(self.organisms[i].energy),
                "alive": bool(self.organisms[i].alive),
                "cellCount": int(self.organisms[i].cell_count),
            })

        any_alive = any(
            self.organisms[i].alive for i in range(self.next_org_id)
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
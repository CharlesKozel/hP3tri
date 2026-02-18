import taichi as ti
import numpy as np
from numpy.typing import NDArray

from simulator.hex_grid import CellType, TerrainType, NEIGHBOR_OFFSETS

ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32)

MAX_ORGANISMS = 64

NEIGHBOR_OFFSETS_TI = ti.Vector.field(2, dtype=ti.i32, shape=6)
for i, (dq, dr) in enumerate(NEIGHBOR_OFFSETS):
    NEIGHBOR_OFFSETS_TI[i] = ti.Vector([dq, dr])


@ti.func
def wrap(q: ti.i32, r: ti.i32, width: ti.i32, height: ti.i32) -> ti.math.ivec2:
    return ti.math.ivec2(q % width, r % height)


@ti.func
def grid_index(q: ti.i32, r: ti.i32, width: ti.i32) -> ti.i32:
    return r * width + q


@ti.data_oriented
class SimulationEngine:
    def __init__(self, width: int, height: int, seed: int = 42) -> None:
        self.width = width
        self.height = height
        self.size = width * height

        self.cell_type = ti.field(dtype=ti.i8, shape=self.size)
        self.organism_id = ti.field(dtype=ti.i32, shape=self.size)
        self.terrain_type = ti.field(dtype=ti.i8, shape=self.size)

        self.org_q = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_r = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_energy = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_alive = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_cell_type = ti.field(dtype=ti.i8, shape=MAX_ORGANISMS)
        self.org_brain_state = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_center_q = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_center_r = ti.field(dtype=ti.i32, shape=MAX_ORGANISMS)
        self.org_count = 0

        self.rng_seed = seed

    def add_organism(
        self,
        q: int, r: int,
        cell_type: int,
        energy: int,
        center_q: int, center_r: int,
    ) -> int:
        idx = self.org_count
        self.org_count += 1
        self.org_q[idx] = q
        self.org_r[idx] = r
        self.org_energy[idx] = energy
        self.org_alive[idx] = 1
        self.org_cell_type[idx] = cell_type
        self.org_brain_state[idx] = 0
        self.org_center_q[idx] = center_q
        self.org_center_r[idx] = center_r

        gidx = r * self.width + q
        self.cell_type[gidx] = cell_type
        self.organism_id[gidx] = idx + 1
        return idx

    @ti.kernel
    def step_brain_eval(
        self,
        org_brain_state: ti.template(),
        org_alive: ti.template(),
        org_count: ti.i32,
    ):
        for i in range(org_count):
            if org_alive[i] == 1:
                org_brain_state[i] = (org_brain_state[i] + 1) % 6

    @ti.kernel
    def step_apply_actions(
        self,
        cell_type_field: ti.template(),
        organism_id_field: ti.template(),
        org_q: ti.template(),
        org_r: ti.template(),
        org_energy: ti.template(),
        org_alive: ti.template(),
        org_cell_type: ti.template(),
        org_brain_state: ti.template(),
        org_center_q: ti.template(),
        org_center_r: ti.template(),
        org_count: ti.i32,
        width: ti.i32,
        height: ti.i32,
    ):
        for i in range(org_count):
            if org_alive[i] == 0:
                continue

            org_energy[i] -= 1
            if org_energy[i] <= 0:
                old_idx = grid_index(org_q[i], org_r[i], width)
                cell_type_field[old_idx] = ti.cast(0, ti.i8)
                organism_id_field[old_idx] = 0
                org_alive[i] = 0
                continue

            direction = org_brain_state[i]
            offset = NEIGHBOR_OFFSETS_TI[direction]
            cq = org_center_q[i]
            cr = org_center_r[i]
            new_q = (cq + offset[0]) % width
            new_r = (cr + offset[1]) % height

            old_idx = grid_index(org_q[i], org_r[i], width)
            new_idx = grid_index(new_q, new_r, width)

            cell_type_field[old_idx] = ti.cast(0, ti.i8)
            organism_id_field[old_idx] = 0

            cell_type_field[new_idx] = org_cell_type[i]
            organism_id_field[new_idx] = i + 1

            org_q[i] = new_q
            org_r[i] = new_r

    def step(self) -> None:
        self.step_brain_eval(
            self.org_brain_state,
            self.org_alive,
            self.org_count,
        )
        self.step_apply_actions(
            self.cell_type,
            self.organism_id,
            self.org_q,
            self.org_r,
            self.org_energy,
            self.org_alive,
            self.org_cell_type,
            self.org_brain_state,
            self.org_center_q,
            self.org_center_r,
            self.org_count,
            self.width,
            self.height,
        )

    def snapshot(self, tick: int) -> dict:
        ct: NDArray[np.int8] = self.cell_type.to_numpy()
        oi: NDArray[np.int32] = self.organism_id.to_numpy()
        tt: NDArray[np.int8] = self.terrain_type.to_numpy()

        tiles: list[dict] = []
        for idx in range(self.size):
            cell = int(ct[idx])
            terrain = int(tt[idx])
            if cell != CellType.EMPTY or terrain != TerrainType.GROUND:
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
        for i in range(self.org_count):
            organisms.append({
                "id": i + 1,
                "energy": int(self.org_energy[i]),
                "alive": bool(self.org_alive[i]),
                "cellCount": 1 if self.org_alive[i] else 0,
            })

        any_alive = any(self.org_alive[i] for i in range(self.org_count))
        status = "RUNNING" if any_alive else "FINISHED"

        return {
            "tick": tick,
            "status": status,
            "grid": {
                "width": self.width,
                "height": self.height,
                "tiles": tiles,
            },
            "organisms": organisms,
        }


def run_simulation(config: dict) -> list[dict]:
    width: int = config.get("width", 32)
    height: int = config.get("height", 32)
    tick_limit: int = config.get("tick_limit", 18)
    seed: int = config.get("seed", 42)

    engine = SimulationEngine(width, height, seed)

    center_q = width // 2
    center_r = height // 2
    start_q = (center_q + NEIGHBOR_OFFSETS[0][0]) % width
    start_r = (center_r + NEIGHBOR_OFFSETS[0][1]) % height

    engine.add_organism(
        q=start_q, r=start_r,
        cell_type=int(CellType.SOFT_TISSUE),
        energy=18,
        center_q=center_q, center_r=center_r,
    )

    replay: list[dict] = [engine.snapshot(0)]

    for tick in range(1, tick_limit + 1):
        engine.step()
        replay.append(engine.snapshot(tick))

    return replay

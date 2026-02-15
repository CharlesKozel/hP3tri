from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class TerrainType(IntEnum):
    GROUND = 0
    WATER = 1
    ROCK = 2
    FERTILE = 3
    TOXIC = 4


class CellType(IntEnum):
    EMPTY = 0
    SKIN = 1
    ARMOR = 2
    MOUTH = 3
    SPIKE = 4
    PHOTOSYNTHETIC = 5
    EYE = 6
    FLAGELLA = 7
    MEMBRANE = 8
    ROOT = 9
    TEETH = 10
    CHEMICAL_SENSOR = 11
    TOUCH_SENSOR = 12
    CILIA = 13
    PSEUDOPOD = 14
    STORAGE_VACUOLE = 15
    REPRODUCTIVE = 16
    SIGNAL_EMITTER = 17
    PIGMENT = 18


# Axial hex neighbor offsets (flat-top orientation)
# Directions: E, NE, NW, W, SW, SE
NEIGHBOR_OFFSETS: list[tuple[int, int]] = [
    (+1, 0),
    (+1, -1),
    (0, -1),
    (-1, 0),
    (-1, +1),
    (0, +1),
]


def wrap(q: int, r: int, width: int, height: int) -> tuple[int, int]:
    return q % width, r % height


def index(q: int, r: int, width: int) -> int:
    return r * width + q


def coords(idx: int, width: int) -> tuple[int, int]:
    r = idx // width
    q = idx % width
    return q, r


def neighbors(
    q: int, r: int, width: int, height: int,
) -> list[tuple[int, int]]:
    return [wrap(q + dq, r + dr, width, height) for dq, dr in NEIGHBOR_OFFSETS]


def _unwrapped_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    s1 = -q1 - r1
    s2 = -q2 - r2
    return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))


def hex_distance(
    q1: int, r1: int, q2: int, r2: int,
    width: int, height: int,
) -> int:
    best = _unwrapped_distance(q1, r1, q2, r2)
    for dq in (-width, 0, width):
        for dr in (-height, 0, height):
            if dq == 0 and dr == 0:
                continue
            d = _unwrapped_distance(q1, r1, q2 + dq, r2 + dr)
            if d < best:
                best = d
    return best


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def cube_round(fq: float, fr: float, fs: float) -> tuple[int, int, int]:
    rq = round(fq)
    rr = round(fr)
    rs = round(fs)

    dq = abs(rq - fq)
    dr = abs(rr - fr)
    ds = abs(rs - fs)

    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    else:
        rs = -rq - rr

    return int(rq), int(rr), int(rs)


def _find_closest_unwrapped(
    q1: int, r1: int, q2: int, r2: int,
    width: int, height: int,
) -> tuple[int, int]:
    best_d = _unwrapped_distance(q1, r1, q2, r2)
    best_q, best_r = q2, r2
    for dq in (-width, 0, width):
        for dr in (-height, 0, height):
            uq, ur = q2 + dq, r2 + dr
            d = _unwrapped_distance(q1, r1, uq, ur)
            if d < best_d:
                best_d = d
                best_q, best_r = uq, ur
    return best_q, best_r


def line_of_sight(
    q1: int, r1: int, q2: int, r2: int,
    width: int, height: int,
) -> list[tuple[int, int]]:
    uq2, ur2 = _find_closest_unwrapped(q1, r1, q2, r2, width, height)
    n = _unwrapped_distance(q1, r1, uq2, ur2)
    if n == 0:
        return [(q1 % width, r1 % height)]

    s1 = -q1 - r1
    s2 = -uq2 - ur2

    results: list[tuple[int, int]] = []
    for i in range(n + 1):
        t = i / n
        fq = lerp(q1, uq2, t)
        fr = lerp(r1, ur2, t)
        fs = lerp(s1, s2, t)
        cq, cr, _ = cube_round(fq, fr, fs)
        results.append(wrap(cq, cr, width, height))

    return results


class HexGrid:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        size = width * height

        self.terrain_type: NDArray[np.int8] = np.zeros(size, dtype=np.int8)
        self.cell_type: NDArray[np.int8] = np.zeros(size, dtype=np.int8)
        self.organism_id: NDArray[np.int32] = np.zeros(size, dtype=np.int32)

    def _index(self, q: int, r: int) -> int:
        return index(q, r, self.width)

    def get_terrain(self, q: int, r: int) -> int:
        return int(self.terrain_type[self._index(q, r)])

    def set_terrain(self, q: int, r: int, terrain: int) -> None:
        self.terrain_type[self._index(q, r)] = terrain

    def get_cell(self, q: int, r: int) -> int:
        return int(self.cell_type[self._index(q, r)])

    def set_cell(
        self, q: int, r: int, cell: int, organism: int = 0,
    ) -> None:
        idx = self._index(q, r)
        self.cell_type[idx] = cell
        self.organism_id[idx] = organism

    def get_organism(self, q: int, r: int) -> int:
        return int(self.organism_id[self._index(q, r)])

    def clear_cell(self, q: int, r: int) -> None:
        idx = self._index(q, r)
        self.cell_type[idx] = CellType.EMPTY
        self.organism_id[idx] = 0

    def to_sparse_dict(self) -> dict:
        tiles = []
        for idx in range(self.width * self.height):
            cell = int(self.cell_type[idx])
            terrain = int(self.terrain_type[idx])
            if cell != CellType.EMPTY or terrain != TerrainType.GROUND:
                q, r = coords(idx, self.width)
                tiles.append({
                    "q": q,
                    "r": r,
                    "terrainType": terrain,
                    "cellType": cell,
                    "organismId": int(self.organism_id[idx]),
                })
        return {
            "width": self.width,
            "height": self.height,
            "tiles": tiles,
        }

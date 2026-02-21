from enum import IntEnum, auto

import taichi as ti



class TerrainType(IntEnum):
    GROUND = auto()


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

OFFSETS_MATRIX = ti.Matrix([
    [+1, 0],
    [+1, -1],
    [0, -1],
    [-1, 0],
    [-1, +1],
    [0, +1],
])

@ti.func
def neighbor_offset(direction: ti.i32) -> ti.math.ivec2:
    return ti.Vector([OFFSETS_MATRIX[direction, 0], OFFSETS_MATRIX[direction, 1]])


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


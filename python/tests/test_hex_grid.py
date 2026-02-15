import numpy as np

from simulator.hex_grid import (
    CellType,
    HexGrid,
    TerrainType,
    coords,
    hex_distance,
    index,
    line_of_sight,
    neighbors,
    wrap,
)

W, H = 16, 16


class TestCoordinateMath:
    def test_index_and_coords_roundtrip(self) -> None:
        for q in range(W):
            for r in range(H):
                idx = index(q, r, W)
                q2, r2 = coords(idx, W)
                assert (q, r) == (q2, r2)

    def test_index_sequential(self) -> None:
        assert index(0, 0, 10) == 0
        assert index(5, 0, 10) == 5
        assert index(0, 1, 10) == 10
        assert index(3, 2, 10) == 23

    def test_wrap(self) -> None:
        assert wrap(0, 0, 10, 10) == (0, 0)
        assert wrap(10, 0, 10, 10) == (0, 0)
        assert wrap(-1, 0, 10, 10) == (9, 0)
        assert wrap(0, -1, 10, 10) == (0, 9)
        assert wrap(11, 12, 10, 10) == (1, 2)


class TestNeighbors:
    def test_interior_has_six_neighbors(self) -> None:
        n = neighbors(5, 5, W, H)
        assert len(n) == 6

    def test_corner_has_six_neighbors_via_wrapping(self) -> None:
        n = neighbors(0, 0, W, H)
        assert len(n) == 6
        assert (W - 1, 0) in n
        assert (0, H - 1) in n

    def test_neighbor_offsets_are_unique(self) -> None:
        n = neighbors(3, 3, W, H)
        assert len(set(n)) == 6

    def test_neighbors_are_distance_one(self) -> None:
        q, r = 5, 5
        for nq, nr in neighbors(q, r, W, H):
            assert hex_distance(q, r, nq, nr, W, H) == 1

    def test_edge_neighbors_are_distance_one(self) -> None:
        for nq, nr in neighbors(0, 0, W, H):
            assert hex_distance(0, 0, nq, nr, W, H) == 1


class TestDistance:
    def test_distance_to_self_is_zero(self) -> None:
        assert hex_distance(3, 3, 3, 3, W, H) == 0

    def test_distance_to_neighbor_is_one(self) -> None:
        assert hex_distance(3, 3, 4, 3, W, H) == 1
        assert hex_distance(3, 3, 3, 2, W, H) == 1

    def test_known_distances(self) -> None:
        assert hex_distance(0, 0, 2, 0, 32, 32) == 2
        assert hex_distance(0, 0, 0, 3, 32, 32) == 3

    def test_distance_is_symmetric(self) -> None:
        assert hex_distance(1, 2, 4, 5, W, H) == hex_distance(4, 5, 1, 2, W, H)

    def test_toroidal_wrapping_shorter_path(self) -> None:
        assert hex_distance(0, 0, 15, 0, W, H) == 1
        assert hex_distance(0, 0, 0, 15, W, H) == 1

    def test_toroidal_vs_direct(self) -> None:
        d_direct = hex_distance(0, 0, 8, 0, 32, 32)
        d_wrapped = hex_distance(0, 0, 8, 0, W, H)
        assert d_direct == 8
        assert d_wrapped == 8


class TestLineOfSight:
    def test_same_point(self) -> None:
        result = line_of_sight(3, 3, 3, 3, W, H)
        assert result == [(3, 3)]

    def test_adjacent(self) -> None:
        result = line_of_sight(0, 0, 1, 0, W, H)
        assert result == [(0, 0), (1, 0)]

    def test_length_matches_distance(self) -> None:
        q1, r1, q2, r2 = 1, 1, 4, 2
        result = line_of_sight(q1, r1, q2, r2, W, H)
        assert len(result) == hex_distance(q1, r1, q2, r2, W, H) + 1

    def test_starts_and_ends_correctly(self) -> None:
        result = line_of_sight(1, 1, 4, 3, W, H)
        assert result[0] == (1, 1)
        assert result[-1] == (4, 3)

    def test_wrapping_line_of_sight(self) -> None:
        result = line_of_sight(0, 0, 15, 0, W, H)
        assert result[0] == (0, 0)
        assert result[-1] == (15, 0)
        assert len(result) == 2

    def test_all_coords_in_bounds(self) -> None:
        result = line_of_sight(1, 1, 14, 14, W, H)
        for q, r in result:
            assert 0 <= q < W
            assert 0 <= r < H


class TestHexGrid:
    def test_creation(self) -> None:
        grid = HexGrid(32, 32)
        assert grid.width == 32
        assert grid.height == 32
        assert len(grid.terrain_type) == 32 * 32
        assert len(grid.cell_type) == 32 * 32
        assert grid.chemical_signals.shape == (32 * 32, 4)

    def test_default_terrain_is_ground(self) -> None:
        grid = HexGrid(8, 8)
        for i in range(64):
            assert grid.terrain_type[i] == TerrainType.GROUND

    def test_default_cells_are_empty(self) -> None:
        grid = HexGrid(8, 8)
        for i in range(64):
            assert grid.cell_type[i] == CellType.EMPTY

    def test_set_and_get_cell(self) -> None:
        grid = HexGrid(16, 16)
        grid.set_cell(5, 3, CellType.SKIN, organism=1)
        assert grid.get_cell(5, 3) == CellType.SKIN
        assert grid.get_organism(5, 3) == 1

    def test_set_and_get_terrain(self) -> None:
        grid = HexGrid(16, 16)
        grid.set_terrain(2, 4, TerrainType.WATER)
        assert grid.get_terrain(2, 4) == TerrainType.WATER

    def test_clear_cell(self) -> None:
        grid = HexGrid(16, 16)
        grid.set_cell(3, 3, CellType.ARMOR, organism=2)
        grid.clear_cell(3, 3)
        assert grid.get_cell(3, 3) == CellType.EMPTY
        assert grid.get_organism(3, 3) == 0


    def test_to_sparse_dict_empty_grid(self) -> None:
        grid = HexGrid(4, 4)
        result = grid.to_sparse_dict()
        assert result["width"] == 4
        assert result["height"] == 4
        assert result["tiles"] == []

    def test_to_sparse_dict_with_cells(self) -> None:
        grid = HexGrid(4, 4)
        grid.set_cell(1, 1, CellType.SKIN, organism=1)
        grid.set_terrain(2, 2, TerrainType.WATER)
        result = grid.to_sparse_dict()
        assert len(result["tiles"]) == 2

import pytest
from simulator.engine import SimulationEngine
from simulator.cell_types import CellType
from simulator.hex_grid import NEIGHBOR_OFFSETS

W, H = 8, 8


def make_engine() -> SimulationEngine:
    return SimulationEngine(W, H, seed=0)


def cell_pos(engine: SimulationEngine, org_id: int) -> tuple[int, int] | None:
    """Return (q, r) of the single cell belonging to org_id, or None."""
    oi = engine.grid.organism_id.to_numpy()
    for idx in range(engine.grid_size):
        if int(oi[idx]) == org_id:
            return idx % engine.width, idx // engine.width
    return None


def run_movement(engine: SimulationEngine, directions: dict[int, int]) -> None:
    """Recompute aggregates, set brain directions, run one movement step."""
    engine.recompute_aggregates()
    for oid, d in directions.items():
        engine.organisms[oid].brain_move_dir = d
    engine.claims.fill(0)
    engine.step_movement(
        engine.grid,
        engine.temp_grid,
        engine.organisms,
        engine.next_org_id,
        engine.width,
        engine.height,
        engine.grid_size,
        engine.claims,
    )


# ── Basic movement ────────────────────────────────────────────────────

def test_single_organism_moves_correct_direction():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    for d in range(6):
        start = cell_pos(engine, oid)
        run_movement(engine, {oid: d})
        end = cell_pos(engine, oid)
        dq, dr = NEIGHBOR_OFFSETS[d]
        expected = ((start[0] + dq) % W, (start[1] + dr) % H)
        assert end == expected, f"dir {d}: expected {expected}, got {end}"


def test_immobile_organism_stays():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.MOUTH), 1000)
    run_movement(engine, {oid: 0})
    assert cell_pos(engine, oid) == (4, 4)


def test_toroidal_wrap():
    engine = make_engine()
    oid = engine.create_organism(0, 0, int(CellType.FLAGELLA), 1000)
    # dir 2 = (0, -1), from r=0 should wrap to r=H-1
    run_movement(engine, {oid: 2})
    assert cell_pos(engine, oid) == (0, H - 1)


# ── Movement point accumulation ──────────────────────────────────────

def test_slow_organism_accumulates_points():
    engine = make_engine()
    # SOFT_TISSUE: locomotion=1, mass=1 → moves every tick
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    run_movement(engine, {oid: 0})
    assert cell_pos(engine, oid) == (5, 4)


def test_heavy_organism_needs_multiple_ticks():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    # Manually increase mass so locomotion(1) < mass(3), needs 3 ticks to accumulate
    engine.recompute_aggregates()
    engine.organisms[oid].total_mass = 3
    engine.organisms[oid].brain_move_dir = 0
    engine.claims.fill(0)
    engine.step_movement(
        engine.grid, engine.temp_grid, engine.organisms,
        engine.next_org_id, engine.width, engine.height, engine.grid_size, engine.claims,
    )
    # After 1 tick: movement_points = 1, need 3 → no move
    assert cell_pos(engine, oid) == (4, 4)
    assert engine.organisms[oid].movement_points == 1

    engine.organisms[oid].brain_move_dir = 0
    engine.claims.fill(0)
    engine.step_movement(
        engine.grid, engine.temp_grid, engine.organisms,
        engine.next_org_id, engine.width, engine.height, engine.grid_size, engine.claims,
    )
    # After 2 ticks: movement_points = 2, need 3 → no move
    assert cell_pos(engine, oid) == (4, 4)
    assert engine.organisms[oid].movement_points == 2

    engine.organisms[oid].brain_move_dir = 0
    engine.claims.fill(0)
    engine.step_movement(
        engine.grid, engine.temp_grid, engine.organisms,
        engine.next_org_id, engine.width, engine.height, engine.grid_size, engine.claims,
    )
    # After 3 ticks: movement_points reached 3, moves, then deducted back to 0
    assert cell_pos(engine, oid) == (5, 4)
    assert engine.organisms[oid].movement_points == 0


def test_movement_points_capped_no_banking():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    # FLAGELLA: locomotion=10, mass=1. Run movement with no direction (brain_move_dir=-1)
    # so it accumulates but doesn't move. Points should cap at mass=1.
    engine.recompute_aggregates()
    engine.organisms[oid].brain_move_dir = -1
    engine.claims.fill(0)
    engine.step_movement(
        engine.grid, engine.temp_grid, engine.organisms,
        engine.next_org_id, engine.width, engine.height, engine.grid_size, engine.claims,
    )
    assert engine.organisms[oid].movement_points == 1  # capped at total_mass

    # Run again with no direction — should stay capped at 1
    engine.organisms[oid].brain_move_dir = -1
    engine.claims.fill(0)
    engine.step_movement(
        engine.grid, engine.temp_grid, engine.organisms,
        engine.next_org_id, engine.width, engine.height, engine.grid_size, engine.claims,
    )
    assert engine.organisms[oid].movement_points == 1


# ── Eligibility checks ───────────────────────────────────────────────

def test_dead_organism_doesnt_move():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    engine.organisms[oid].alive = 0
    run_movement(engine, {oid: 0})
    assert cell_pos(engine, oid) == (4, 4)


def test_zero_energy_doesnt_move():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 0)
    run_movement(engine, {oid: 0})
    assert cell_pos(engine, oid) == (4, 4)


def test_negative_brain_dir_doesnt_move():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    run_movement(engine, {oid: -1})
    assert cell_pos(engine, oid) == (4, 4)


# ── Blocking ──────────────────────────────────────────────────────────

def test_blocked_by_foreign_organism():
    engine = make_engine()
    mover = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    _blocker = engine.create_organism(5, 4, int(CellType.MOUTH), 1000)
    # dir 0 = (+1, 0) → mover tries to go to (5, 4) which is occupied
    run_movement(engine, {mover: 0})
    assert cell_pos(engine, mover) == (4, 4)


def test_blocked_by_dead_cell():
    engine = make_engine()
    mover = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    # Place a dead cell (cell_type set, organism_id = 0) at destination
    dest_idx = 4 * W + 5  # (5, 4)
    engine.grid[dest_idx].cell_type = int(CellType.SOFT_TISSUE)
    engine.grid[dest_idx].organism_id = 0
    run_movement(engine, {mover: 0})
    assert cell_pos(engine, mover) == (4, 4)


# ── Priority — claim conflicts ────────────────────────────────────────

def test_higher_speed_wins_claim():
    engine = make_engine()
    # Both target (4, 4): fast from (3, 4) dir 0, slow from (4, 3) dir 5
    fast = engine.create_organism(3, 4, int(CellType.FLAGELLA), 1000)  # speed 10
    slow = engine.create_organism(4, 3, int(CellType.SOFT_TISSUE), 1000)  # speed 1
    run_movement(engine, {fast: 0, slow: 5})
    assert cell_pos(engine, fast) == (4, 4), "faster organism should win the cell"
    assert cell_pos(engine, slow) == (4, 3), "slower organism should stay"


def test_equal_speed_younger_wins():
    engine = make_engine()
    # Both FLAGELLA (same speed), both targeting (4, 4)
    older = engine.create_organism(3, 4, int(CellType.FLAGELLA), 1000)  # lower org_id
    younger = engine.create_organism(4, 3, int(CellType.FLAGELLA), 1000)  # higher org_id
    run_movement(engine, {older: 0, younger: 5})
    assert cell_pos(engine, younger) == (4, 4), "younger (higher id) should win tiebreak"
    assert cell_pos(engine, older) == (3, 4), "older should stay"


def test_loser_retains_movement_points():
    engine = make_engine()
    fast = engine.create_organism(3, 4, int(CellType.FLAGELLA), 1000)
    slow = engine.create_organism(4, 3, int(CellType.SOFT_TISSUE), 1000)
    run_movement(engine, {fast: 0, slow: 5})
    # slow lost the claim — should still have its movement_points
    assert engine.organisms[slow].movement_points > 0
    # slow's energy should NOT have been deducted
    assert engine.organisms[slow].energy == 1000


# ── Energy accounting ─────────────────────────────────────────────────

def test_successful_move_deducts_energy():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    run_movement(engine, {oid: 0})
    # energy cost = total_mass = 1 (single FLAGELLA cell)
    assert engine.organisms[oid].energy == 999


def test_successful_move_deducts_movement_points():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    run_movement(engine, {oid: 0})
    # movement_points should be 0 after moving (accumulated mass, then deducted mass)
    assert engine.organisms[oid].movement_points == 0


def test_failed_move_no_energy_deduction():
    engine = make_engine()
    mover = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    _blocker = engine.create_organism(5, 4, int(CellType.MOUTH), 1000)
    run_movement(engine, {mover: 0})
    assert engine.organisms[mover].energy == 1000


# ── Edge case: vacating organisms ─────────────────────────────────────

def test_organism_blocked_by_cell_being_vacated():
    """
    A moves toward B's current position, B is also moving away.
    Current implementation checks the original grid, so A should be blocked
    even though B is leaving. This documents current behavior.
    Hard to avoid this logic because movement may fail, future upgrade to change.
    """
    engine = make_engine()
    a = engine.create_organism(3, 4, int(CellType.FLAGELLA), 1000)
    b = engine.create_organism(4, 4, int(CellType.FLAGELLA), 1000)
    # A moves right into B's cell (dir 0), B moves right away (dir 0)
    run_movement(engine, {a: 0, b: 0})
    # B should move to (5, 4)
    assert cell_pos(engine, b) == (5, 4)
    # A is blocked by B's original position — documents current behavior
    # If this assertion fails in the future (i.e. A moves to (4,4)), that means
    # the implementation was improved to allow moving into vacated cells.
    assert cell_pos(engine, a) == (3, 4)
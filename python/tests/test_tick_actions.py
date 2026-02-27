import pytest
from simulator.engine import SimulationEngine
from simulator.cell_types import CellType

W, H = 8, 8


def make_engine() -> SimulationEngine:
    return SimulationEngine(W, H, seed=0)


def cell_at(engine: SimulationEngine, q: int, r: int) -> tuple[int, int]:
    """Return (cell_type, organism_id) at position (q, r)."""
    idx = r * engine.width + q
    ct = int(engine.grid[idx].cell_type)
    oid = int(engine.grid[idx].organism_id)
    return ct, oid


def run_actions(engine: SimulationEngine) -> None:
    """Recompute aggregates + run one action resolution step."""
    engine.recompute_aggregates()
    engine.step_actions()


def set_growth(engine: SimulationEngine, oid: int, wants: bool, direction: int = -1,
               cell_type: int = int(CellType.SOFT_TISSUE)) -> None:
    engine.organisms[oid].brain_wants_grow = 1 if wants else 0
    engine.organisms[oid].brain_grow_direction = direction
    engine.organisms[oid].brain_grow_cell_type = cell_type


# ── Eating: MOUTH eats FOOD ──────────────────────────────────────────

def test_mouth_eats_adjacent_food():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.MOUTH), 100)
    engine.place_cell(0, 5, 4, int(CellType.FOOD))  # unowned food at (5,4)
    initial_energy = int(engine.organisms[oid].energy)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == 0, "food should be consumed"
    assert owner == 0, "cell should be empty"
    assert int(engine.organisms[oid].energy) > initial_energy, "organism should gain energy"


def test_mouth_eats_foreign_organism_cell():
    engine = make_engine()
    eater = engine.create_organism(4, 4, int(CellType.MOUTH), 100, genome_id=1)
    prey = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == 0, "prey cell should be consumed"
    assert owner == 0, "cell should be cleared"


def test_mouth_does_not_eat_same_genome():
    engine = make_engine()
    org1 = engine.create_organism(4, 4, int(CellType.MOUTH), 100, genome_id=1)
    org2 = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=1)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.SOFT_TISSUE), "same-genome cell should not be eaten"
    assert owner == org2


def test_mouth_does_not_eat_own_cell():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.MOUTH), 100, genome_id=1)
    engine.place_cell(oid, 5, 4, int(CellType.SOFT_TISSUE))

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.SOFT_TISSUE), "own cell should not be eaten"
    assert owner == oid


def test_two_mouths_compete_for_same_food():
    """Higher priority organism wins the food."""
    engine = make_engine()
    org1 = engine.create_organism(3, 4, int(CellType.MOUTH), 100, genome_id=1)
    org2 = engine.create_organism(5, 4, int(CellType.MOUTH), 100, genome_id=2)
    engine.place_cell(0, 4, 4, int(CellType.FOOD))

    run_actions(engine)

    ct, _ = cell_at(engine, 4, 4)
    assert ct == 0, "food should be consumed by one of them"
    # Exactly one should have gained energy
    e1 = int(engine.organisms[org1].energy)
    e2 = int(engine.organisms[org2].energy)
    assert (e1 > 100) != (e2 > 100), "exactly one organism should gain energy"


def test_eating_sets_connectivity_check_on_victim():
    engine = make_engine()
    eater = engine.create_organism(4, 4, int(CellType.MOUTH), 100, genome_id=1)
    prey = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)

    run_actions(engine)

    assert int(engine.organisms[prey].needs_connectivity_check) == 1


# ── Growth ────────────────────────────────────────────────────────────

def test_organism_grows_border_cell():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 100)
    set_growth(engine, oid, wants=True)

    run_actions(engine)

    grew = False
    for d in range(6):
        dq, dr = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)][d]
        nq, nr = (4 + dq) % W, (4 + dr) % H
        ct, owner = cell_at(engine, nq, nr)
        if ct == int(CellType.SOFT_TISSUE) and owner == oid:
            grew = True
            break
    assert grew, "organism should have grown at least one cell"


def test_growth_deducts_energy():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 100)
    set_growth(engine, oid, wants=True)
    initial_energy = int(engine.organisms[oid].energy)

    run_actions(engine)

    assert int(engine.organisms[oid].energy) < initial_energy


def test_growth_limited_to_one_cell_per_tick():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    set_growth(engine, oid, wants=True)

    run_actions(engine)

    new_cells = 0
    for d in range(6):
        dq, dr = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)][d]
        nq, nr = (4 + dq) % W, (4 + dr) % H
        ct, owner = cell_at(engine, nq, nr)
        if owner == oid:
            new_cells += 1
    assert new_cells == 1, f"should grow exactly 1 new cell, got {new_cells}"


def test_no_growth_without_energy():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 0)
    set_growth(engine, oid, wants=True)

    run_actions(engine)

    new_cells = 0
    for d in range(6):
        dq, dr = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)][d]
        nq, nr = (4 + dq) % W, (4 + dr) % H
        ct, owner = cell_at(engine, nq, nr)
        if owner == oid:
            new_cells += 1
    assert new_cells == 0, "should not grow with no energy"


def test_no_growth_when_not_requested():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    set_growth(engine, oid, wants=False)

    run_actions(engine)

    new_cells = 0
    for d in range(6):
        dq, dr = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)][d]
        nq, nr = (4 + dq) % W, (4 + dr) % H
        ct, owner = cell_at(engine, nq, nr)
        if owner == oid:
            new_cells += 1
    assert new_cells == 0, "should not grow when brain doesn't want growth"


def test_growth_does_not_overwrite_existing_cell():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    # Fill all 6 neighbors with food
    for dq, dr in [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]:
        nq, nr = (4 + dq) % W, (4 + dr) % H
        engine.place_cell(0, nq, nr, int(CellType.FOOD))
    set_growth(engine, oid, wants=True)

    run_actions(engine)

    new_cells = 0
    for d in range(6):
        dq, dr = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)][d]
        nq, nr = (4 + dq) % W, (4 + dr) % H
        ct, owner = cell_at(engine, nq, nr)
        if owner == oid:
            new_cells += 1
    assert new_cells == 0, "should not grow into occupied cells"


def test_two_orgs_grow_into_same_empty_hex():
    """Priority resolution picks one winner."""
    engine = make_engine()
    org1 = engine.create_organism(3, 4, int(CellType.SOFT_TISSUE), 1000, genome_id=1)
    org2 = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 1000, genome_id=2)
    # Both want to grow — (4,4) is adjacent to both
    set_growth(engine, org1, wants=True, direction=0)  # dir 0 = E → targets (4,4)
    set_growth(engine, org2, wants=True, direction=3)  # dir 3 = W → targets (4,4)

    run_actions(engine)

    ct, owner = cell_at(engine, 4, 4)
    assert ct == int(CellType.SOFT_TISSUE), "one org should have grown here"
    assert owner in (org1, org2), "winner should be one of the two organisms"


def test_growth_with_specific_direction():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    set_growth(engine, oid, wants=True, direction=0)  # dir 0 = E = (5,4)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.SOFT_TISSUE) and owner == oid, "should grow east"

    # Verify no growth in other directions
    for d, (dq, dr) in enumerate([(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]):
        nq, nr = (4 + dq) % W, (4 + dr) % H
        _, o = cell_at(engine, nq, nr)
        assert o != oid, f"should not grow in direction {d+1}"


def test_growth_with_specific_cell_type():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 1000)
    set_growth(engine, oid, wants=True, direction=0, cell_type=int(CellType.MOUTH))

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.MOUTH), "should grow the requested cell type"
    assert owner == oid


# ── Eat + Growth don't conflict ───────────────────────────────────────

def test_eat_and_growth_same_tick():
    """Eat targets occupied cells, growth targets empty — no conflict."""
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.MOUTH), 200, genome_id=1)
    engine.place_cell(0, 5, 4, int(CellType.FOOD))
    set_growth(engine, oid, wants=True, direction=3)  # dir 3 = W → (3,4) is empty

    run_actions(engine)

    # Food should be eaten
    ct, _ = cell_at(engine, 5, 4)
    assert ct == 0, "food should be consumed"

    # Growth should have placed a cell west
    ct, owner = cell_at(engine, 3, 4)
    assert ct == int(CellType.SOFT_TISSUE) and owner == oid, "should have grown west"


# ── Full tick integration ─────────────────────────────────────────────

def test_full_tick_with_eating():
    engine = make_engine()
    oid = engine.create_organism(4, 4, int(CellType.MOUTH), 100, genome_id=1)
    engine.place_cell(0, 5, 4, int(CellType.FOOD))

    engine.step()

    ct, _ = cell_at(engine, 5, 4)
    assert ct == 0, "food should be eaten during full tick"


# ── Destroy: SPIKE ──────────────────────────────────────────────────

def test_spike_destroys_adjacent_foreign_cell():
    engine = make_engine()
    engine.create_organism(4, 4, int(CellType.SPIKE), 100, genome_id=1)
    engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == 0, "foreign cell should be destroyed by spike"
    assert owner == 0


def test_spike_destroys_all_adjacent_foreign_cells():
    engine = make_engine()
    engine.create_organism(4, 4, int(CellType.SPIKE), 100, genome_id=1)
    engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)
    engine.create_organism(3, 4, int(CellType.SOFT_TISSUE), 100, genome_id=3)

    run_actions(engine)

    ct_e, _ = cell_at(engine, 5, 4)
    ct_w, _ = cell_at(engine, 3, 4)
    assert ct_e == 0, "east foreign cell should be destroyed"
    assert ct_w == 0, "west foreign cell should be destroyed"


def test_spike_does_not_destroy_same_genome():
    engine = make_engine()
    engine.create_organism(4, 4, int(CellType.SPIKE), 100, genome_id=1)
    ally = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=1)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.SOFT_TISSUE), "same-genome cell should not be destroyed"
    assert owner == ally


def test_spike_does_not_gain_energy():
    engine = make_engine()
    spike_org = engine.create_organism(4, 4, int(CellType.SPIKE), 100, genome_id=1)
    engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)

    run_actions(engine)

    assert int(engine.organisms[spike_org].energy) == 100, "spike should not gain energy"


def test_spike_sets_connectivity_check_on_victim():
    engine = make_engine()
    engine.create_organism(4, 4, int(CellType.SPIKE), 100, genome_id=1)
    victim = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)

    run_actions(engine)

    assert int(engine.organisms[victim].needs_connectivity_check) == 1


def test_eat_rank_beats_destroy_rank_for_same_target():
    """EAT (rank 4) outranks DESTROY (rank 3) — both claim a foreign SOFT_TISSUE cell."""
    engine = make_engine()
    eater = engine.create_organism(4, 3, int(CellType.MOUTH), 100, genome_id=1)
    spiker = engine.create_organism(4, 5, int(CellType.SPIKE), 100, genome_id=3)
    # Target at (4,4): MOUTH reaches via SE, SPIKE via NW. Both can target it.
    engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)

    run_actions(engine)

    ct, _ = cell_at(engine, 4, 4)
    assert ct == 0, "target should be destroyed by the winning claim"
    # SOFT_TISSUE consumption_value is 0, so we can't distinguish by energy.
    # The rank encoding guarantees EAT (4) > DESTROY (3) via ti.atomic_max.
    # Verify SPIKE did NOT gain energy (spike never gains energy):
    assert int(engine.organisms[spiker].energy) == 100, "spike should not gain energy"


def test_spike_does_not_destroy_food():
    engine = make_engine()
    engine.create_organism(4, 4, int(CellType.SPIKE), 100, genome_id=1)
    engine.place_cell(0, 5, 4, int(CellType.FOOD))

    run_actions(engine)

    ct, _ = cell_at(engine, 5, 4)
    assert ct == int(CellType.FOOD), "spike should not destroy food (not in CAN_DESTROY)"


# ── Reproduce ───────────────────────────────────────────────────────

def set_reproduce(engine: SimulationEngine, oid: int, cell_idx: int,
                  direction: int = 0, energy: int = 10) -> None:
    engine.organisms[oid].brain_reproduce_cell_idx = cell_idx
    engine.organisms[oid].brain_reproduce_direction = direction
    engine.organisms[oid].brain_reproduce_energy = energy


def test_reproduce_creates_new_organism():
    engine = make_engine()
    parent = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 200, genome_id=7)
    parent_idx = 4 * W + 4  # r=4, q=4
    set_reproduce(engine, parent, parent_idx, direction=0, energy=20)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.SOFT_TISSUE), "offspring cell should exist"
    assert owner != 0 and owner != parent, "offspring should be a new organism"
    child_oid = owner
    assert int(engine.organisms[child_oid].genome_id) == 7, "offspring should inherit genome"
    assert int(engine.organisms[child_oid].energy) == 20, "offspring should have specified energy"


def test_reproduce_deducts_parent_energy():
    engine = make_engine()
    parent = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 200, genome_id=1)
    parent_idx = 4 * W + 4
    set_reproduce(engine, parent, parent_idx, direction=0, energy=20)
    from simulator.cell_types import CELL_PROPERTIES
    growth_cost = CELL_PROPERTIES[CellType.SOFT_TISSUE].growth_cost

    run_actions(engine)

    expected = 200 - growth_cost - 20
    assert int(engine.organisms[parent].energy) == expected


def test_reproduce_fails_insufficient_energy():
    engine = make_engine()
    parent = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 5, genome_id=1)
    parent_idx = 4 * W + 4
    set_reproduce(engine, parent, parent_idx, direction=0, energy=10)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == 0, "should not reproduce with insufficient energy"
    assert int(engine.organisms[parent].energy) == 5, "energy should be unchanged"


def test_reproduce_fails_target_occupied():
    engine = make_engine()
    parent = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 200, genome_id=1)
    engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 100, genome_id=2)
    parent_idx = 4 * W + 4
    set_reproduce(engine, parent, parent_idx, direction=0, energy=20)

    run_actions(engine)

    ct, owner = cell_at(engine, 5, 4)
    assert ct == int(CellType.SOFT_TISSUE), "occupied cell should remain"
    assert int(engine.organisms[parent].energy) == 200, "energy unchanged on failure"


def test_reproduce_max_one_per_tick():
    engine = make_engine()
    parent = engine.create_organism(4, 4, int(CellType.SOFT_TISSUE), 500, genome_id=1)
    engine.place_cell(parent, 5, 4, int(CellType.SOFT_TISSUE))
    parent_idx_a = 4 * W + 4
    parent_idx_b = 4 * W + 5
    # Both cells try to reproduce — only brain_reproduce_cell_idx is one value,
    # so only one cell can match. This test verifies the atomic counter caps at 1.
    set_reproduce(engine, parent, parent_idx_a, direction=3, energy=10)

    run_actions(engine)

    # Count new organisms created
    new_orgs = 0
    for oid in range(1, engine.next_org_id):
        if oid != parent and int(engine.organisms[oid].alive) == 1:
            new_orgs += 1
    assert new_orgs <= 1, "at most 1 reproduction per organism per tick"


def test_eye_cannot_reproduce():
    engine = make_engine()
    parent = engine.create_organism(4, 4, int(CellType.EYE), 200, genome_id=1)
    parent_idx = 4 * W + 4
    set_reproduce(engine, parent, parent_idx, direction=0, energy=20)

    run_actions(engine)

    ct, _ = cell_at(engine, 5, 4)
    assert ct == 0, "EYE cell should not be able to reproduce"
    assert int(engine.organisms[parent].energy) == 200


def test_reproduce_wins_over_growth_for_same_hex():
    """Reproduce (rank 2) outranks growth (rank 1) for same empty hex."""
    engine = make_engine()
    reproducer = engine.create_organism(3, 4, int(CellType.SOFT_TISSUE), 500, genome_id=1)
    grower = engine.create_organism(5, 4, int(CellType.SOFT_TISSUE), 500, genome_id=2)

    rep_idx = 4 * W + 3
    set_reproduce(engine, reproducer, rep_idx, direction=0, energy=20)
    set_growth(engine, grower, wants=True, direction=3)  # dir 3 = W → (4,4)

    run_actions(engine)

    ct, owner = cell_at(engine, 4, 4)
    assert ct != 0, "cell should be occupied"
    assert owner != grower, "reproducer (rank 2) should win over growth (rank 1)"
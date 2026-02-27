from simulator.cell_types import CellType
from simulator.engine import SimulationEngine


def run_simulation(config: dict) -> list[dict]:
    width: int = config.get("width", 32)
    height: int = config.get("height", 32)
    tick_limit: int = config.get("tick_limit", 100)
    seed: int = config.get("seed", 42)

    engine = SimulationEngine(width, height, seed)

    # --- Test 1: Head-on collision (same speed, tiebreak by org_id) ---
    # Two FLAGELLA (speed=10) on same row, heading toward each other.
    # org_id=1 → dir 1 → (+1,-1), org_id=2 → dir 4 → (-1,+1) (opposite)
    # They approach on the q+1/r-1 diagonal. Younger (org_id=2) should win tiebreak.
    engine.create_organism(  # org 1: dir (1 % 6)=1 → (+1,-1)
        seed_q=5,
        seed_r=16,
        seed_cell_type=int(CellType.FLAGELLA),
        starting_energy=10_000,
        genome_id=1,
    )
    engine.create_organism(  # org 2: dir (2 % 6)=2 → (0,-1)
        seed_q=8,
        seed_r=16,
        seed_cell_type=int(CellType.FLAGELLA),
        starting_energy=10_000,
        genome_id=1,
    )

    # --- Test 2: Speed priority (FLAGELLA vs SOFT_TISSUE toward same area) ---
    # FLAGELLA (speed=10) vs SOFT_TISSUE (speed=1). Both move every tick but
    # FLAGELLA has higher speed_score so it wins priority on contested cells.
    engine.create_organism(  # org 3: dir (3 % 6)=3 → (-1,0)
        seed_q=20,
        seed_r=8,
        seed_cell_type=int(CellType.FLAGELLA),
        starting_energy=10_000,
        genome_id=2,
    )
    engine.create_organism(  # org 4: dir (4 % 6)=4 → (-1,+1)
        seed_q=18,
        seed_r=7,
        seed_cell_type=int(CellType.SOFT_TISSUE),
        starting_energy=10_000,
        genome_id=2,
    )

    # --- Test 3: Mobile blocked by immobile ---
    # FLAGELLA heading directly into a stationary MOUTH.
    # org 5: dir (5 % 6)=5 → (0,+1). MOUTH at (10, 26) blocks path.
    engine.create_organism(  # org 5: FLAGELLA moving down
        seed_q=10,
        seed_r=24,
        seed_cell_type=int(CellType.FLAGELLA),
        starting_energy=10_000,
        genome_id=3,
    )
    engine.create_organism(  # org 6: stationary MOUTH in the way
        seed_q=10,
        seed_r=26,
        seed_cell_type=int(CellType.MOUTH),
        starting_energy=10_000,
        genome_id=3,
    )

    replay: list[dict] = []
    engine.recompute_aggregates()
    replay.append(engine.snapshot())

    for tick in range(tick_limit):
        engine.step()
        replay.append(engine.snapshot())

    return replay

if __name__ == '__main__':
    run_simulation({})
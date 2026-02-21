from simulator.cell_types import CellType
from simulator.engine import SimulationEngine


def run_simulation(config: dict) -> list[dict]:
    width: int = config.get("width", 32)
    height: int = config.get("height", 32)
    tick_limit: int = config.get("tick_limit", 100)
    seed: int = config.get("seed", 42)

    engine = SimulationEngine(width, height, seed)

    center_q = width // 2
    center_r = height // 2

    engine.create_organism(
        seed_q=0,
        seed_r=0,
        seed_cell_type=int(CellType.FLAGELLA),
        starting_energy=10_000,
    )

    replay: list[dict] = []
    engine.recompute_aggregates()
    replay.append(engine.snapshot())

    for tick in range(tick_limit):
        engine.step()
        replay.append(engine.snapshot())

    return replay

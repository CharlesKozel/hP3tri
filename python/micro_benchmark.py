"""
Micro-benchmark for quick GPU performance testing.
Single grid size, measures throughput and identifies bottlenecks.
"""

import os
import sys
import time

# Set CUDA and disable verbose logging
os.environ.setdefault("TAICHI_ARCH", "cuda")

import taichi as ti
import numpy as np

# Initialize with reduced logging
ti.init(arch=ti.cuda, default_ip=ti.i32, default_fp=ti.f32, log_level=ti.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.engine import SimulationEngine
from simulator.cell_types import CellType


def create_organisms(engine: SimulationEngine, num_organisms: int):
    """Create scattered organisms for testing."""
    width, height = engine.width, engine.height
    rng = engine.rng
    cell_types = [CellType.FLAGELLA, CellType.SOFT_TISSUE, CellType.MOUTH, CellType.PHOTOSYNTHETIC]

    positions = set()
    created = 0
    for _ in range(num_organisms * 10):
        if created >= num_organisms:
            break
        q = int(rng.integers(0, width))
        r = int(rng.integers(0, height))
        if (q, r) not in positions:
            positions.add((q, r))
            engine.create_organism(
                seed_q=q, seed_r=r,
                seed_cell_type=int(rng.choice(cell_types)),
                starting_energy=1000,
                genome_id=created % 10,
                seed_direction=int(rng.integers(0, 6)),
            )
            created += 1
    return created


def benchmark(width: int, height: int, num_organisms: int, warmup: int = 20, measure: int = 100):
    """Run benchmark and return timing data."""
    print(f"Grid: {width}x{height} ({width*height} cells), Organisms: {num_organisms}")

    engine = SimulationEngine(width, height, seed=42)
    actual = create_organisms(engine, num_organisms)
    print(f"Created {actual} organisms")

    engine.recompute_aggregates()

    # Warmup - compile kernels
    print(f"Warmup ({warmup} ticks)...")
    for _ in range(warmup):
        engine.step()
    ti.sync()

    # Measure total throughput
    print(f"Measuring ({measure} ticks)...")
    start = time.perf_counter()
    for _ in range(measure):
        engine.step()
    ti.sync()
    elapsed = time.perf_counter() - start

    tps = measure / elapsed
    ms_per_tick = elapsed / measure * 1000

    print(f"\n{'='*50}")
    print(f"RESULTS: {width}x{height} grid, {actual} organisms")
    print(f"{'='*50}")
    print(f"  Ticks per second: {tps:.1f}")
    print(f"  Time per tick:    {ms_per_tick:.3f} ms")
    print(f"  Total time:       {elapsed:.2f} s")

    # Component timing
    print(f"\nComponent timing (20 ticks):")
    print("-" * 50)

    components = [
        ("recompute_aggregates", engine.recompute_aggregates),
        ("apply_resources", engine.apply_resources),
        ("step_sensors", engine.step_sensors),
        ("step_brains", engine.step_brains),
        ("process_movement", engine.process_movement),
        ("step_actions", engine.step_actions),
        ("death_and_disconnect", engine.process_death_and_disconnection),
        ("increment_ages", engine.increment_ages),
    ]

    times = {name: [] for name, _ in components}
    total_times = []

    for _ in range(20):
        ti.sync()
        tick_start = time.perf_counter()

        for name, func in components:
            ti.sync()
            t0 = time.perf_counter()
            func()
            ti.sync()
            times[name].append(time.perf_counter() - t0)

        ti.sync()
        total_times.append(time.perf_counter() - tick_start)
        engine.tick_count += 1

    avg_total = np.mean(total_times) * 1000
    for name, _ in components:
        avg = np.mean(times[name]) * 1000
        pct = avg / avg_total * 100
        print(f"  {name:30s}: {avg:8.3f} ms ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s}: {avg_total:8.3f} ms")

    return {
        "ticks_per_second": tps,
        "ms_per_tick": ms_per_tick,
        "component_times": {n: np.mean(times[n]) * 1000 for n, _ in components},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256, help="Grid size (NxN)")
    parser.add_argument("--orgs", type=int, default=1000, help="Number of organisms")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup ticks")
    parser.add_argument("--measure", type=int, default=100, help="Measurement ticks")
    args = parser.parse_args()

    print(f"Taichi backend: {os.environ.get('TAICHI_ARCH', 'cuda')}")
    print(f"Testing GPU utilization with RTX 3090\n")

    benchmark(args.size, args.size, args.orgs, args.warmup, args.measure)


if __name__ == "__main__":
    main()

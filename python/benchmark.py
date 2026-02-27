"""
GPU Benchmark for hP3tri Simulation Engine

Measures simulation performance across different grid sizes and organism counts
to identify bottlenecks and measure GPU utilization.
"""

import os
import sys
import time
import argparse

# Set CUDA as default backend before importing Taichi
if "TAICHI_ARCH" not in os.environ:
    os.environ["TAICHI_ARCH"] = "cuda"

import taichi as ti
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.engine import SimulationEngine
from simulator.cell_types import CellType


def create_benchmark_organisms(engine: SimulationEngine, num_organisms: int, strategy: str = "random"):
    """
    Populate the grid with organisms for benchmarking.

    Strategies:
    - random: scattered single-cell organisms
    - clustered: multi-cell organisms in clusters
    - dense: maximum density organisms
    """
    width = engine.width
    height = engine.height
    rng = engine.rng

    cell_types = [
        CellType.FLAGELLA,
        CellType.SOFT_TISSUE,
        CellType.MOUTH,
        CellType.PHOTOSYNTHETIC,
        CellType.EYE,
    ]

    if strategy == "random":
        # Scatter single-cell organisms randomly
        positions_used = set()
        created = 0
        attempts = 0
        max_attempts = num_organisms * 10

        while created < num_organisms and attempts < max_attempts:
            q = int(rng.integers(0, width))
            r = int(rng.integers(0, height))
            if (q, r) not in positions_used:
                ct = int(rng.choice(cell_types))
                engine.create_organism(
                    seed_q=q,
                    seed_r=r,
                    seed_cell_type=ct,
                    starting_energy=1000,
                    genome_id=created % 10,  # 10 different genomes
                    seed_direction=int(rng.integers(0, 6)),
                )
                positions_used.add((q, r))
                created += 1
            attempts += 1

    elif strategy == "clustered":
        # Create multi-cell organisms (5-10 cells each)
        created = 0
        cluster_size = 7
        spacing = max(10, int(np.sqrt(width * height / num_organisms)))

        for start_q in range(0, width, spacing):
            for start_r in range(0, height, spacing):
                if created >= num_organisms:
                    break

                ct = int(rng.choice(cell_types))
                org_id = engine.create_organism(
                    seed_q=start_q,
                    seed_r=start_r,
                    seed_cell_type=ct,
                    starting_energy=5000,
                    genome_id=created % 10,
                )

                # Add surrounding cells to create a cluster
                from simulator.hex_grid import OFFSETS
                for i, (dq, dr) in enumerate(OFFSETS):
                    if i >= cluster_size - 1:
                        break
                    nq = (start_q + dq) % width
                    nr = (start_r + dr) % height
                    cell_ct = int(rng.choice(cell_types))
                    engine.place_cell(org_id, nq, nr, cell_ct, direction=i)

                created += 1

    elif strategy == "dense":
        # Fill grid as densely as possible
        created = 0
        for q in range(0, width, 2):
            for r in range(0, height, 2):
                if created >= num_organisms:
                    break
                ct = int(rng.choice(cell_types))
                engine.create_organism(
                    seed_q=q,
                    seed_r=r,
                    seed_cell_type=ct,
                    starting_energy=1000,
                    genome_id=created % 10,
                )
                created += 1
            if created >= num_organisms:
                break

    return created


def benchmark_tick_components(engine: SimulationEngine, warmup_ticks: int = 5, measure_ticks: int = 20):
    """
    Measure time spent in each simulation step component.
    """
    # Warmup to compile kernels
    for _ in range(warmup_ticks):
        engine.step()

    ti.sync()

    # Measure each component
    results = {
        "recompute_aggregates": [],
        "apply_resources": [],
        "step_sensors": [],
        "step_brains": [],
        "process_movement": [],
        "step_actions": [],
        "process_death_and_disconnection": [],
        "increment_ages": [],
        "total_step": [],
    }

    for _ in range(measure_ticks):
        # Total step time
        ti.sync()
        total_start = time.perf_counter()

        # Recompute aggregates
        ti.sync()
        t0 = time.perf_counter()
        engine.recompute_aggregates()
        ti.sync()
        results["recompute_aggregates"].append(time.perf_counter() - t0)

        # Apply resources
        ti.sync()
        t0 = time.perf_counter()
        engine.apply_resources()
        ti.sync()
        results["apply_resources"].append(time.perf_counter() - t0)

        # Sensors
        ti.sync()
        t0 = time.perf_counter()
        engine.step_sensors()
        ti.sync()
        results["step_sensors"].append(time.perf_counter() - t0)

        # Brains (CPU-bound)
        ti.sync()
        t0 = time.perf_counter()
        engine.step_brains()
        ti.sync()
        results["step_brains"].append(time.perf_counter() - t0)

        # Movement
        ti.sync()
        t0 = time.perf_counter()
        engine.process_movement()
        ti.sync()
        results["process_movement"].append(time.perf_counter() - t0)

        # Actions
        ti.sync()
        t0 = time.perf_counter()
        engine.step_actions()
        ti.sync()
        results["step_actions"].append(time.perf_counter() - t0)

        # Death and disconnection
        ti.sync()
        t0 = time.perf_counter()
        engine.process_death_and_disconnection()
        ti.sync()
        results["process_death_and_disconnection"].append(time.perf_counter() - t0)

        # Increment ages
        ti.sync()
        t0 = time.perf_counter()
        engine.increment_ages()
        ti.sync()
        results["increment_ages"].append(time.perf_counter() - t0)

        ti.sync()
        results["total_step"].append(time.perf_counter() - total_start)

        engine.tick_count += 1

    # Compute averages
    avg_results = {k: np.mean(v) * 1000 for k, v in results.items()}  # Convert to ms
    return avg_results


def benchmark_throughput(engine: SimulationEngine, warmup_ticks: int = 10, measure_ticks: int = 100):
    """
    Measure overall simulation throughput (ticks per second).
    """
    # Warmup
    for _ in range(warmup_ticks):
        engine.step()

    ti.sync()

    # Measure
    start = time.perf_counter()
    for _ in range(measure_ticks):
        engine.step()
    ti.sync()
    elapsed = time.perf_counter() - start

    ticks_per_second = measure_ticks / elapsed
    ms_per_tick = elapsed / measure_ticks * 1000

    return {
        "ticks_per_second": ticks_per_second,
        "ms_per_tick": ms_per_tick,
        "total_time_sec": elapsed,
        "ticks_measured": measure_ticks,
    }


def run_benchmark_suite(
    grid_sizes: list[tuple[int, int]],
    organism_counts: list[int],
    strategy: str = "random",
    warmup_ticks: int = 10,
    measure_ticks: int = 50,
):
    """
    Run comprehensive benchmark across grid sizes and organism counts.
    """
    results = []

    for width, height in grid_sizes:
        for num_orgs in organism_counts:
            # Skip impossible combinations
            max_orgs = (width * height) // 4
            if num_orgs > max_orgs:
                print(f"Skipping {width}x{height} with {num_orgs} orgs (max={max_orgs})")
                continue

            print(f"\n{'='*60}")
            print(f"Benchmarking: {width}x{height} grid, {num_orgs} organisms")
            print(f"{'='*60}")

            # Create engine
            engine = SimulationEngine(width, height, seed=42)

            # Populate
            actual_orgs = create_benchmark_organisms(engine, num_orgs, strategy)
            print(f"Created {actual_orgs} organisms")

            # Initial aggregates
            engine.recompute_aggregates()

            # Component timing
            print("\nComponent timing (ms per tick):")
            print("-" * 40)
            component_times = benchmark_tick_components(engine, warmup_ticks=5, measure_ticks=20)
            for component, time_ms in component_times.items():
                pct = time_ms / component_times["total_step"] * 100 if component != "total_step" else 100
                print(f"  {component:35s}: {time_ms:8.3f} ms ({pct:5.1f}%)")

            # Throughput
            print("\nThroughput measurement:")
            print("-" * 40)
            throughput = benchmark_throughput(engine, warmup_ticks, measure_ticks)
            print(f"  Ticks per second: {throughput['ticks_per_second']:.1f}")
            print(f"  Time per tick:    {throughput['ms_per_tick']:.3f} ms")

            results.append({
                "grid_size": (width, height),
                "grid_cells": width * height,
                "num_organisms": actual_orgs,
                "strategy": strategy,
                "component_times_ms": component_times,
                "throughput": throughput,
            })

            # Clean up Taichi fields (force garbage collection)
            del engine

    return results


def print_summary(results: list[dict]):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Grid Size':>12} {'Cells':>10} {'Orgs':>8} {'Ticks/s':>10} {'ms/tick':>10} {'Brain %':>10}")
    print("-" * 80)

    for r in results:
        w, h = r["grid_size"]
        brain_pct = r["component_times_ms"]["step_brains"] / r["component_times_ms"]["total_step"] * 100
        print(f"{w}x{h:>6} {r['grid_cells']:>10} {r['num_organisms']:>8} "
              f"{r['throughput']['ticks_per_second']:>10.1f} "
              f"{r['throughput']['ms_per_tick']:>10.3f} "
              f"{brain_pct:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark hP3tri simulation engine")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (smaller grids)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (all grid sizes)")
    parser.add_argument("--stress", action="store_true", help="Stress test (large grids)")
    parser.add_argument("--grid", type=str, help="Single grid size WxH (e.g., 512x512)")
    parser.add_argument("--orgs", type=int, help="Number of organisms")
    parser.add_argument("--strategy", choices=["random", "clustered", "dense"], default="random")
    parser.add_argument("--arch", type=str, default=None, help="Taichi arch (cuda, cpu, vulkan)")

    args = parser.parse_args()

    # Set architecture if specified
    if args.arch:
        os.environ["TAICHI_ARCH"] = args.arch

    arch = os.environ.get("TAICHI_ARCH", "cuda")
    print(f"Taichi backend: {arch}")

    # Reinitialize Taichi with correct arch
    ti.reset()
    arch_map = {"cuda": ti.cuda, "cpu": ti.cpu, "vulkan": ti.vulkan, "metal": ti.metal}
    ti.init(arch=arch_map.get(arch, ti.cuda), default_ip=ti.i32, default_fp=ti.f32)

    if args.grid:
        # Single configuration
        w, h = map(int, args.grid.split("x"))
        orgs = args.orgs or (w * h) // 8
        results = run_benchmark_suite(
            grid_sizes=[(w, h)],
            organism_counts=[orgs],
            strategy=args.strategy,
        )
    elif args.quick:
        results = run_benchmark_suite(
            grid_sizes=[(64, 64), (128, 128), (256, 256)],
            organism_counts=[100, 500, 1000],
            strategy=args.strategy,
        )
    elif args.full:
        results = run_benchmark_suite(
            grid_sizes=[(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
            organism_counts=[100, 500, 1000, 2000, 5000],
            strategy=args.strategy,
        )
    elif args.stress:
        results = run_benchmark_suite(
            grid_sizes=[(512, 512), (1024, 1024), (2048, 2048)],
            organism_counts=[1000, 5000, 10000],
            strategy=args.strategy,
        )
    else:
        # Default moderate benchmark
        results = run_benchmark_suite(
            grid_sizes=[(128, 128), (256, 256), (512, 512)],
            organism_counts=[500, 1000, 2000],
            strategy=args.strategy,
        )

    print_summary(results)

    return results


if __name__ == "__main__":
    main()

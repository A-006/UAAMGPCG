#!/usr/bin/env python3
"""
Karman Vortex Street — Solver Benchmark Suite

Orchestrates the full pipeline:
  1. Generate INPUT files for all solvers/grids
  2. Run lfm_2d simulations
  3. Generate comparison figures

Usage:
  python run_all.py              # Full pipeline
  python run_all.py --inputs     # Only generate INPUT files
  python run_all.py --run        # Only run simulations
  python run_all.py --plot       # Only generate plots from existing data
  python run_all.py --plot 128x32  # Plot for specific grid
  python run_all.py --grid 128x32  # Run only one grid resolution
"""

import argparse, sys, os

# Ensure config module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *


def main():
    parser = argparse.ArgumentParser(
        description="Karman Vortex Street — Solver Benchmark Suite")
    parser.add_argument("--inputs", action="store_true", help="Only generate INPUT files")
    parser.add_argument("--run", action="store_true", help="Only run simulations")
    parser.add_argument("--plot", action="store_true", help="Only generate plots")
    parser.add_argument("--grid", type=str, default=None,
                        help="Grid resolution (e.g., 128x32, 256x64)")
    parser.add_argument("--t-end", type=float, default=None,
                        help="Override simulation end time")
    args = parser.parse_args()

    params = KarmanParams()
    if args.t_end is not None:
        params.t_end = args.t_end
        print(f"Using t_end = {params.t_end}")

    # Determine which steps to run
    do_inputs = args.inputs or not (args.run or args.plot)
    do_run = args.run or not (args.inputs or args.plot)
    do_plot = args.plot or not (args.inputs or args.run)

    # Optionally restrict grid
    if args.grid:
        import config
        config.GRIDS = [g for g in config.GRIDS if g.label == args.grid]
        if not config.GRIDS:
            print(f"Unknown grid: {args.grid}. Available: 128x32, 256x64")
            return 1

    # ── Step 1: Generate INPUTs ──
    if do_inputs:
        print("=" * 60)
        print("  Step 1: Generating INPUT files")
        print("=" * 60)
        from generate_inputs import generate_all
        generate_all(params)
        print()

    # ── Step 2: Run simulations ──
    if do_run:
        print("=" * 60)
        print("  Step 2: Running simulations")
        print("=" * 60)
        from run_simulations import run_all
        results = run_all(params)
        print()

    # ── Step 3: Generate plots ──
    if do_plot:
        print("=" * 60)
        print("  Step 3: Generating comparison figures")
        print("=" * 60)
        from plot_results import plot_all
        plot_all(args.grid)
        print()

    print("Done!")


if __name__ == "__main__":
    main()

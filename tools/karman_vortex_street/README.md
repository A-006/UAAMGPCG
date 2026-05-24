# Karman Vortex Street — Solver Benchmark Suite

Automated comparison of pressure solvers for the Kármán vortex street problem.

## Quick Start

```bash
# Run the full pipeline (128×32 + 256×64 grids, 4 solvers)
python run_all.py

# Fast run — single grid only
python run_all.py --grid 128x32 --t-end 1.0

# Generate plots from existing data
python run_all.py --plot
```

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `generate_inputs.py` | Create INPUT files for each solver/grid combination |
| 2 | `run_simulations.py` | Execute `lfm_2d` and collect timing/accuracy data |
| 3 | `plot_results.py` | Render comparison figures from VTK output |

## Configuration

Edit `config.py` to customize solvers, grid sizes, and physical parameters.

## Output

```
tools/karman_vortex_street/
├── inputs/         # Generated INPUT files
├── output/         # VTK data + results.json per solver
│   ├── cg/
│   ├── pcg_gmg/
│   ├── pcg_amg/
│   ├── pcg_uaamg/
│   └── results.json
└── figures/        # Comparison images
    ├── vorticity_comparison_128_32.png
    ├── vorticity_comparison_256_64.png
    ├── velocity_comparison_*.png
    ├── divergence_history_*.png
    └── performance_table.png
```

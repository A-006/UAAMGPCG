# UAAMGPCG — 2D Matrix-Free Fluid Simulation with AMG-Preconditioned CG

Reference implementation from Chapter 5 of the Leapfrog Flow Maps paper. Includes standalone Ax=b solvers and a full 2D LFM fluid simulator with four pluggable pressure solvers.

## Paper Reference

**Leapfrog Flow Maps** — SIGGRAPH 2025  
*Matrix-Free AMGPCG Solver on GPU for Real-Time Fluid Simulation*

Key citations:
- Deng et al. 2023 — *Neural Flow Maps* (LFM's direct predecessor)
- Nabizadeh et al. 2022 — *Covector Fluids* (impulse-based fluid simulation)
- Shao et al. 2022 — Matrix-free UAAMG CPU SIMD implementation
- Stuben 2001 — *A Review of Algebraic Multigrid*
- Shewchuk 1994 — *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*

## Project Structure

```
UAAMGPCG/
├── include/                         # Headers (mirrors src/ layout)
│   ├── lfm/                         # LFM simulation headers
│   │   ├── config.h                 #   Config + Solver enum
│   │   ├── grid.h                   #   MAC staggered grid
│   │   ├── advection.h              #   Interpolation + RK2 advection
│   │   ├── poisson_jacobi.h         #   Jacobi solver (MAC grid)
│   │   ├── poisson_rbgs.h           #   RBGS solver (MAC grid)
│   │   ├── poisson_cg.h             #   CG solver (MAC grid)
│   │   ├── poisson_pcg.h            #   PCG solver (MAC grid, V-Cycle preconditioned)
│   │   ├── pressure.h               #   Pressure projection
│   │   ├── boundary.h               #   Boundary conditions
│   │   └── vtk_io.h                 #   VTK output
│   └── solver/
│       └── jacobi.h                 #   Generic Jacobi iteration template
├── src/
│   ├── main.cpp                     # Entry point + LFM time step
│   ├── lfm/                         # LFM implementations (mirrors include/lfm/)
│   │   ├── grid.cpp
│   │   ├── advection.cpp
│   │   ├── poisson_jacobi.cpp       #   Jacobi — simple, slow, baseline
│   │   ├── poisson_rbgs.cpp         #   RBGS — ~2× faster than Jacobi
│   │   ├── poisson_cg.cpp           #   CG — O(√κ) convergence
│   │   ├── poisson_pcg.cpp          #   PCG — V-Cycle preconditioned, grid-independent
│   │   ├── pressure.cpp             #   Solver dispatch
│   │   ├── boundary.cpp
│   │   └── vtk_io.cpp
│   └── solver/                      # Standalone Ax=b solvers (educational)
│       ├── jacobi.cpp / rbgs.cpp / cg.cpp / vcycle.cpp / amgpcg.cpp
│       └── CMakeLists.txt
└── test/                            # 56 tests (41 unit + 15 integration)
    ├── test_utils.h
    ├── test_jacobi.cpp / test_cg.cpp / test_vcycle.cpp / test_amgpcg.cpp
    ├── test_lfm.cpp                 # Grid, interpolation, BC, Jacobi tests
    └── test_integration.cpp         # Multi-solver, multi-step simulation tests
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requires C++17 and CMake >= 3.10.

## LFM Fluid Simulation

### Run

```bash
# Karman vortex street (Jacobi solver)
./build/src/lfm_2d karman 128 100 jacobi

# With CG solver (faster convergence)
./build/src/lfm_2d karman 128 100 cg

# Smoke buoyancy (PCG solver)
./build/src/lfm_2d smoke 128 100 pcg

# Usage: lfm_2d [karman|smoke] [NX] [t_end] [jacobi|rbgs|cg|pcg]
```

Output VTK files in `output_karman/` or `output_smoke/` — open in ParaView.

### Available Solvers

| Solver | Flag | Method | Speed | Notes |
|--------|------|--------|-------|-------|
| Jacobi | `jacobi` | Matrix-free Jacobi iteration | Slow | O(N²) convergence, baseline |
| RBGS | `rbgs` | Red-Black Gauss-Seidel | ~2× Jacobi | Gauss-Seidel convergence, parallelizable |
| CG | `cg` | Conjugate Gradient | Fast | O(√κ) convergence |
| PCG | `pcg` | V-Cycle preconditioned CG | Fastest | Near grid-independent (~10 iterations) |

All solvers handle non-uniform dx/dy, Neumann BC, and solid cells. All are **matrix-free** — no sparse matrix is ever assembled.

### How Jacobi is Wired In

The standalone solver `src/solver/jacobi.cpp` exposes a generic Jacobi template in `include/solver/jacobi.h`. The LFM pressure solver (`src/lfm/poisson_jacobi.cpp`) adapts this to the MAC grid by providing:
- A matrix-free `Ax` lambda (5-point Laplacian with Neumann BC and solid handling)
- A per-cell `D⁻¹` lambda (effective diagonal)
- Zero-mean RHS and per-iteration pressure mean subtraction (Neumann null space)

## Standalone Solvers

Educational solvers for the Poisson equation on [0,1]² with Dirichlet BC:

```bash
./build/src/solver/jacobi 64     # Jacobi iteration
./build/src/solver/rbgs 64       # RBGS smoother
./build/src/solver/cg 64         # Conjugate Gradient
./build/src/solver/vcycle 64     # V-Cycle / Full Multigrid
./build/src/solver/amgpcg 64     # AMG-preconditioned CG
```

## Test Suite

```bash
cmake --build build
./build/test/test_jacobi        # 4 tests — solver accuracy + convergence
./build/test/test_cg            # 5 tests — matvec, dot, CG accuracy, speed
./build/test/test_vcycle        # 8 tests — hierarchy, restrict/prolong, FMG
./build/test/test_amgpcg        # 5 tests — PCG accuracy, grid-independence
./build/test/test_lfm           # 18 tests — grid, interpolation, BC, Jacobi
./build/test/test_integration   # 12 tests — multi-solver, multi-step LFM sim
```

### Expected Results

| Test Suite | Tests | Coverage |
|-----------|-------|----------|
| test_jacobi | 4 | Convergence, discretization error, BC enforcement, monotonicity |
| test_cg | 5 | matvec, dot product, sin/poly solves, iteration count |
| test_vcycle | 8 | Hierarchy levels, restrict, prolongate, RBGS, FMG accuracy |
| test_amgpcg | 5 | Sin/poly accuracy, CG comparison, grid independence |
| test_lfm | 18 | Grid dims, divergence, interpolation, clamp, solid, Jacobi |
| test_integration | 12 | Uniform flow × 4 solvers, cylinder × 2, buoyancy × 4, NaN checks |

## Algorithm Summary

```
LFM Time Step:
  1. External forces (buoyancy)
  2. RK2 semi-Lagrangian advection (backtrace + interpolate)
  3. Pressure projection:
     a. rhs = ∇·ũ / Δt
     b. Solve ∇²p = rhs  (Jacobi / RBGS / CG / PCG)
     c. u = ũ - Δt·∇p
```

All matrix-vector products use the 5-point Laplacian stencil computed on-the-fly from neighbor values — no sparse matrix is ever assembled.

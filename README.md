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

## GPU 3D UAAMGPCG Solver (CUDA)

FP32 matrix-free AMG-preconditioned CG for 3D Poisson equation on GPU. Implements the full optimization pipeline from Sun et al. SIGGRAPH 2025 Section 5.

### Build (CUDA)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run Benchmarks

```bash
# Double-precision benchmark
./build/src/solver/cuda/test_paper_bench

# Float-precision benchmark (FP32 vs FP64 comparison)
./build/src/solver/cuda/test_paper_bench_f

# Correctness tests
./build/src/solver/cuda/test_cuda_uaamg_3d    # GPU vs CPU V-cycle + PCG
./build/src/solver/cuda/test_cuda_uaamg       # 2D GPU correctness
./build/src/solver/cuda/test_apply_twice      # Multi-call consistency
```

### Performance (RTX 3090, 256×128×128 = 4.2M cells, V(1,1), 40 iters)

| Precision | Solver | Time | Per Iter | Speedup vs FP64-baseline |
|-----------|--------|------|----------|---------------------------|
| FP64 | solve() (original) | 224 ms | 5.6 ms | 1.00x |
| FP64 | solve_optimized() | 189 ms | 4.7 ms | 1.18x |
| **FP32** | **solve_optimized()** | **94 ms** | **2.35 ms** | **2.38x** |

vs Paper (256×128×128, RTX 4090, V(1,1), float): 0.81ms/iter.
Our normalized to RTX 4090: ~1.02ms/iter. Gap ~1.26x.

### Optimization Pipeline (Paper Section 5.3-5.4)

Implemented optimizations, from simplest to most advanced:

| # | Optimization | Technique | Impact | File |
|---|-------------|-----------|--------|------|
| 1 | Shared-memory tiled RBGS | 8³ tile + 1-cell halo in shared memory. 6 neighbor reads from SMEM instead of GMEM. | ~6x fewer global reads | `_3d_opt.cu` |
| 2 | Aggregated RBGS+Restrict | Smooth + residual + 8-to-1 restriction in single kernel via atomicAdd. No x write+re-read. | Saves 2 global mem ops/cell | `_3d_opt.cu` |
| 3 | Aggregated Prolong+RBGS | Prolongation correction + post-smooth in single kernel (up-sweep). | Saves 2 global mem ops/cell | `_3d_opt.cu` |
| 4 | Fused coarsest sweeps | All 20 sweeps in 1 kernel launch. b cached in register. | Saves 19 launches/V-cycle | `_3d_opt.cu` |
| 5 | Redundant zero-x eliminated | All coarse x zeroed once upfront, not per-level. | Saves 1 launch/level | `_3d_opt.cu` |
| 6 | cudaMemsetAsync | Replace custom zero kernels with HW-accelerated memset. | ~5µs saved/launch | `_3d_opt.cu` |
| 7 | inv_diag precomputed | Precompute 1.0/diag. Most cells use multiply instead of divide. | 1 division → 1 multiply | `_3d_opt.cu` |
| 8 | Fused dot products | dot(p,Ap) in matvec kernel, dot(r,r) in axpy kernel. Shared-mem tiled matvec (8³+halo). | 12→8 launches/iter | `_3d_opt.cu` |
| 9 | Tile classification + fast path | Pre-compute which tiles are interior (all-fluid). Fast-path kernel skips all solid-mask checks + eff_d adjustments. | 12 branches + 1 div saved/cell for >90% cells | `_opt_f.cu` |
| 10 | FP32 precision | float (4B) instead of double (8B). Halves memory traffic and register pressure. | ~2x speedup | All `_f` files |

### Kernel Launch Budget (V-cycle, 4 levels, V(1,1))

```
Original (separate kernels):  14 launches (4 pre×2 + 4 restrict + 4 prolong + 4 post×2 + 1 coarse×20)
Optimized (aggregated):        7 launches (3 aggregated↓ + 1 coarse + 3 aggregated↑)
Optimized (memset):            4 launches (3 aggregated↓ + 1 coarse + 3 aggregated↑, zero via memset)
```

### Architecture

```
CudaGrid3Df (FP32) / CudaGrid3D (FP64)
    ↓
CudaUAAMGPreconditioner3Df::apply_optimized()
    ├── classify_trimmed_kernel_f    # Mark interior tiles
    ├── rbgs_restrict_aggregated_kernel_f  # Down-sweep: smooth + restrict
    ├── rbgs_coarsest_kernel_f       # Coarsest: 20 sweeps fused
    └── prolong_rbgs_aggregated_kernel_f   # Up-sweep: prolong + smooth
    ↓
CudaPCG3Df::solve_optimized()
    ├── matvec_tiled_dot_kernel_f    # Matvec + fused dot(p,Ap)
    ├── axpy_dot_kernel_f            # AXPY + fused dot(r,r)
    └── apply_optimized()            # V-cycle preconditioner
```

### Key Design Decisions

- **Matrix-free**: No sparse matrix storage. 7-point Laplacian computed on-the-fly from neighbor values.
- **Shared-memory tiling**: All kernels use 8³ tiles + 1-cell face halos. 6 neighbor reads hit SMEM.
- **double atomicAdd**: Used in restrict to accumulate 8-to-1 residual. Safe on CC≥6.0 (Pascal+).
- **RBGS red-black ordering**: (i+j+k) parity. Red pass → `__syncthreads` → black pass. Tile boundaries use previous-sweep values (same convergence as separate-kernel).
- **SOA-style**: Interleaved x/b/solid arrays accessed via column-major indexing for coalesced loads.

### Future Optimizations (not yet implemented)

| Optimization | Expected Impact | Description |
|-------------|----------------|-------------|
| SOA 5-channel data layout | 1.3-1.5x | Pack coefficients into 5 channels (1 bool + 4 float) for coalesced access |
| CUB global scan for dot reduction | 1.1-1.2x | GPU-side parallel reduction instead of host-side sequential accumulation |
| Texture memory for solid mask | ~1.1x | Use read-only texture cache for solid lookups |
| Multi-stream V-cycles | 1.2-1.5x | Pipeline multiple V-cycles across streams for concurrent execution |
| Warp-level primitives | ~1.1x | Use `__shfl_sync` for halo exchange instead of `__syncthreads` |

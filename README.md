# UAAMGPCG — 2D Matrix-Free AMG-Preconditioned Conjugate Gradient

2D reference implementation of the **UAAMG (Unsmoothed Aggregation Algebraic Multigrid)** solver from Chapter 5 of the Leapfrog Flow Maps paper.

## Paper Reference

**Leapfrog Flow Maps** — SIGGRAPH 2025  
*Matrix-Free AMGPCG Solver on GPU for Real-Time Fluid Simulation*

Key references:
- Deng et al. 2023 — *Neural Flow Maps* (LFM's direct predecessor)
- Nabizadeh et al. 2022 — *Covector Fluids* (impulse-based fluid simulation)
- Shao et al. 2022 — Matrix-free UAAMG CPU SIMD implementation
- Stuben 2001 — *A Review of Algebraic Multigrid*
- Shewchuk 1994 — *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*

## Project Structure

```
UAAMGPCG/
├── CMakeLists.txt
├── solvers/                     # Ax=b solving methods
│   ├── jacobi.cpp               # Jacobi iteration
│   ├── rbgs.cpp                 # Red-Black Gauss-Seidel smoother
│   ├── cg.cpp                   # Conjugate Gradient
│   ├── vcycle.cpp               # V-Cycle / Full Multigrid (FMG)
│   └── amgpcg.cpp               # AMG-preconditioned CG (core algorithm)
└── src/                         # 2D fluid simulation
    └── lfm_2d.cpp               # LFM fluid sim (uses Jacobi for pressure)
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requires C++17 and CMake >= 3.10.

## Unit Tests

Each solver is a standalone executable that validates against a known analytical solution.

### Jacobi (simplest Ax=b solver)

```bash
./build/solvers/jacobi 64
```

Solves $-\Delta u = f$ on $[0,1]^2$ with $u = \sin(\pi x)\sin(\pi y)$ as the manufactured solution.
Verifies that the maximum error against the true solution is below $10^{-3}$ after convergence.

### RBGS (Red-Black Gauss-Seidel)

```bash
./build/solvers/rbgs 64
```

Compares RBGS convergence speed vs Jacobi on the same Poisson problem.
RBGS converges ~2x faster and does not require a separate output array.

### CG (Conjugate Gradient)

```bash
./build/solvers/cg 64
```

Solves $-\Delta u = 2x(1-x) + 2y(1-y)$ with $u = x(1-x)y(1-y)$.
CG converges in ~$O(\sqrt{\kappa})$ iterations vs Jacobi's $O(\kappa)$ — orders of magnitude fewer.

### V-Cycle / FMG (Full Multigrid)

```bash
./build/solvers/vcycle 64
```

Demonstrates the matrix-free multigrid hierarchy (restriction, prolongation, Galerkin coarse-grid operator).
Prints the level hierarchy and verifies that convergence is nearly independent of grid size.

### AMGPCG (core paper algorithm)

```bash
./build/solvers/amgpcg 64
```

AMG-preconditioned CG — uses one V-Cycle per CG iteration as the preconditioner $z = M^{-1}r$.
Converges in ~10 iterations regardless of grid size (vs ~400+ for unpreconditioned CG at N=128).

### LFM 2D Fluid Simulation

```bash
./build/src/lfm_2d karman 128 100    # Karman vortex street
./build/src/lfm_2d smoke 64 100      # Buoyancy-driven smoke
```

Full Leapfrog Flow Map time-stepping with MAC staggered grid, RK2 semi-Lagrangian advection,
Jacobi pressure projection, and VTK output. Open `output_*/frame_*.vtk` in ParaView.

## Expected Test Results (N=64)

| Solver   | Iterations | Error      | Notes                        |
|----------|-----------|------------|------------------------------|
| Jacobi   | ~3800     | < 1e-3     | Slow, grid-dependent         |
| RBGS     | ~2000     | < 1e-3     | ~2x faster than Jacobi       |
| CG       | ~58       | < 1e-12    | Much faster convergence      |
| FMG      | 1 cycle   | ~0.01      | Near grid-independent        |
| AMGPCG   | ~10       | < 1e-3     | Optimal — grid-independent   |

## Algorithm Summary

```
PCG with UAAMG Preconditioner (one CG iteration):

  1. V-Cycle(z = r):                 // Preconditioner M^{-1}
     a. Pre-smooth:   RBGS(x, b)     // Eliminate high-frequency error
     b. Restrict:     r_c = R * r    // 4-to-1 averaging (2D)
     c. Recurse to coarser level
     d. Prolongate:   x += P * x_c   // Constant interpolation + correction
     e. Post-smooth:  RBGS(x, b)

  2. Ap = matrix_free_matvec(p)      // 5-point Laplacian stencil
  3. alpha = (r . z) / (p . Ap)      // Step length along search direction
  4. x += alpha * p                  // Update solution
  5. r -= alpha * Ap                 // Update residual
  6. beta = (r_new . z_new) / (r_old . z_old)
  7. p = z + beta * p                // A-conjugate search direction
```

All operations are **matrix-free** — the Laplacian is computed on-the-fly from neighbor values.
No sparse matrix is ever assembled.

# Semi-structured matrix-free multigrid Poisson solver (2D + 3D, CPU + GPU)

A self-contained, header-only reproduction of

> Mengdi Wang, Yuchen Sun, Bo Zhu,
> *"Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive
> Octrees"*, arXiv:2604.18886 (code: `wang-mengdi/Cirrus-amg`).

"Semi-structured" here means a **stack of uniform level grids plus a leaf/refined
mask** (SPGrid / tile style): each level is a structured uniform grid; adaptivity
lives only in which cells are leaves. The data is stored in flat SoA arrays so it
can later be mapped onto GPU tiles.

## What is implemented

| Paper ingredient | File |
|---|---|
| Semi-structured adaptive grid, 2:1 graded, leaf/refined/outside | `adaptive_grid_2d.h` |
| Matrix-free compact operator (1 diag + neg-face coeffs, eq.4) and composite finite-volume Laplacian with **conservative, symmetric, consistent T-junction coupling** (eq.9–13) | `poisson_operator_2d.h` |
| **Algebraically consistent coarsening** as octree-aggregation Galerkin `A_c = PᵀAP` (constant prolongation `P`, `R=αPᵀ` — the paper's eq.5–7), plus a geometric (non-conservative) baseline | `multigrid_2d.h` |
| Matrix-free **Algorithm 3** (`Coarsen`) compact-coefficient coarsening | `coarsen_alg3_2d.h` |
| V/μ-cycle with `β=2` over-correction + symmetric Gauss-Seidel | `multigrid_2d.h` |
| PCG (Algorithm 1), zero-mean for pure-Neumann | `pcg_2d.h` |
| Circle/star SDF + narrow-band refinement (the sphere/star setup) | `sdf_2d.h` |
| Dense direct solver (independent ground truth) | `dense_solve.h` |

### T-junction operator

The cross-level coupling at a coarse/fine face is assembled as a **rank-1 PSD
stiffness** `K = κ_face · w wᵀ` with `w_coarse = 1`, `w_fine = −1/m` over the `m`
fine cells sharing the face (`κ_face = 4/3` in 2D). This is simultaneously:

* **conservative** — the interface fluxes sum to zero (eq.11, `f1+f3=f4`);
* **symmetric / SPD** — `K` is positive semi-definite (so PCG is valid);
* **consistent** — the normal gradient is transversely averaged (centred), which
  is what restores near-2nd-order accuracy on adaptive grids.

It introduces small positive fine–fine off-diagonals (the `+1/3` terms); these are
part of the PSD rank-1 block and do not break SPD.

## Verification (`test/semistruct/`)

Run after building (`cmake --build build --target test_operator_2d
test_laplacian_accuracy_2d test_poisson_convergence_2d`):

* **`test_operator_2d`** — matrix-free operator == independent textbook 5-point
  Laplacian; symmetry and discrete conservation `A·1=0` on uniform / two-level /
  narrow-band / **random-graded** grids; SPD with a Dirichlet side; explicit
  T-junction `κ = 2/3` symmetric; **matrix-free Algorithm 3 == Galerkin `PᵀAP/2`
  to machine precision**.
* **`test_laplacian_accuracy_2d`** — MMS, volume-weighted RMS: 2nd order on
  uniform grids; ~2nd order in the interior of adaptive grids. (All-cell RMS is
  dominated by O(1) pointwise truncation at the O(n) T-junction cells — a known
  FV property; the meaningful 2nd order is on the solution.)
* **`test_poisson_convergence_2d`** — solution accuracy ~2nd order (uniform and
  adaptive); **grid-independent** PCG iteration count; algebraically-consistent
  coarsening converges faster than geometric on adaptive grids (Fig.14 contrast);
  matrix-free PCG == dense direct solve; robust on random adaptive grids.

## 3D (`*_3d.h`)

Full mirror: 4 coefficients, 8 children, 6 faces. Key difference — in 3D the
coefficients carry an `h` dimension (`κ_same = h_l`; in 2D it was a dimensionless
`1` only by coincidence); the coarse/fine face is a rank-1 stiffness over `m=4`
fine cells with `κ_face = (4/3)h_l`. Same verification suite (`test_*_3d`):
uniform 2.00 order, narrow-band sphere solution 1.23→1.96, grid-independent
iters 4–5, Alg-3 == Galerkin to machine precision.

## Cut-cell (`PoissonOperator{2,3}D::buildCut`)

Solid obstacle via per-face fluid **area fractions** (eq.4 generalisation):
Neumann solid + Dirichlet air, fluid-only DOF set. The operator carries its own
active-DOF list (`node_*`) so the multigrid coarsens over fluid cells only.
This is where algebraic consistency matters most: `test_cutcell_{2,3}d` reproduce
the paper's Fig.14 — Galerkin coarsening stays grid-independent on cut cells
(2D: 5,6,6 PCG iters; 3D: 5,6,6) while geometric coarsening degrades (2D:
20,31,48; 3D: 13,18,21).

## GPU (`test/cuda/test_semistruct_gpu_2d.cu`)

The algebraically-consistent hierarchy is built on the host; the hot loop —
PCG with a multigrid V-cycle preconditioner — runs entirely on the GPU
(CSR SpMV, weighted-Jacobi smoothing, scatter/gather restrict/prolong through
the aggregation map). Validated GPU solution == CPU solution to ~1e-13.

## Still open (future work)

* GPU tile-fused single-kernel path (the paper's peak-throughput 8³-tile
  shared-memory kernels) — the current GPU port is CSR-based, not tile-fused.
* 3D cut-cell combined with T-junctions in one grid (cut-cell currently targets
  uniform grids; adaptivity and cut-cell are each validated separately).
* Free-surface / moving obstacles (paper Sec.4.6).

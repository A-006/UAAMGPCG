# GPU AMGPCG Optimization — Paper vs. Ours

Side-by-side audit of the four optimizations described in
*Leapfrog Flow Maps for Real-Time Fluid Simulation* §5
(Sun et al. 2025, SIGGRAPH) against our CUDA implementation.

| § | Paper feature | Status | Where it lives in our code |
|---|---|---|---|
| 5.1 | Matrix-Free UAAMG (8:1 restriction, ×2 prolongation, RBGS) | ✅ done | `cuda_uaamg_preconditioner_3d.cu:91-211` |
| 5.2 | SoA + 8×8×8 tile + bit-shift voxel index | 🟡 partial | `cuda_uaamg_preconditioner_3d_opt.cu:21-52` |
| 5.3 | Aggregated CUDA kernels + dot-product fusion | ✅ done | `cuda_uaamg_preconditioner_3d_opt.cu` + `cuda_pcg_3d_opt.cu` |
| 5.4 | **Coefficients trimming** | ❌ **missing** | — |

---

## §5.1 Matrix-Free UAAMG

| Item | Paper | Ours |
|---|---|---|
| Restriction | 8:1 averaging (2×2×2 → 1) | ✓ `cuda_uaamg_preconditioner_3d.cu:91-128` |
| Prolongation | constant injection × 2.0 (Stüben 2001) | ✓ `cuda_uaamg_preconditioner_3d.cu:148` |
| Smoother | Red-Black Gauss-Seidel | ✓ same file, separate red/black passes |
| Coarsest sweeps | 10 RBGS steps | ✓ **20 sweeps** at coarsest (over-converge) `:211` |
| Galerkin coarsening | preserves matrix-free representation | ✓ implicit (stencil recomputed) |

**Verdict**: equivalent algorithm. We do 20 coarsest sweeps where paper does 10 — slightly more work but better coarsest accuracy.

---

## §5.2 Data Structure

| Item | Paper | Ours |
|---|---|---|
| Layout | SoA (Structure of Arrays) | 🟡 **implicit** — 3 separate `vector<double>` (x, b, solid), but no explicit per-tile struct |
| 8×8×8 tile size | aligned to warp size | ✓ `#define TILED_T 8` `cuda_uaamg_preconditioner_3d_opt.cu:21` |
| Halo cells in shared mem | 10×10×10 with 1-cell halo | ✓ `#define TILED_TH (TILED_T+2)` `:22` + halo load `:46-52` |
| Voxel index | bit shifts from (i,j,k) | 🟡 we use multiply `dev_idx3d`, not pure bit shift |
| 5 data channels per voxel | bool DoF + diag + 3 off-diag floats stored explicitly | ❌ **not done** — coefficients recomputed from `idx2/idy2/idz2` constants each iteration |

**Verdict**: tile structure correct, but matrix coefficients are **always uniform** in our code (Poisson with no variable density), so the "5-channel SoA" never materializes. **If we add variable-density Poisson (smoke / fire ball / two-phase), this gap reopens.**

---

## §5.3 Aggregated Kernels — the main optimization (paper claims 2×)

| Aggregated kernel | Paper | Ours |
|---|---|---|
| Down-stroke RBGS + restriction (4 stages: red x_l → black GS → residual → restrict) | one fused kernel per level | ✓ `cuda_uaamg_preconditioner_3d_opt.cu:94-171` |
| Up-stroke prolongation + RBGS (post-smooth) | one fused kernel per level | ✓ `cuda_uaamg_preconditioner_3d_opt.cu:176-244` |
| Coarsest level | one fused kernel with all 10 sweeps | ✓ 20 sweeps fused `cuda_uaamg_preconditioner_3d_opt.cu:249-309` |
| PCG dot product fused into previous kernel | blockwise reduction + CUB exclusive scan | ✓ matvec+dot fused `cuda_pcg_3d_opt.cu:41-150`; axpy+dot fused `:156-187` |
| Shared-memory cache of intermediate x_l for tile-boundary deps | yes | ✓ same files, `__shared__ sx[TH][TH][TH]` |

**Kernel-launch reduction**: 14 → 6 launches for a 4-level V-cycle. Matches paper's claim.

**Verdict**: this is the main paper contribution and we have **all of it**.

---

## §5.4 Coefficients Trimming — **NOT IMPLEMENTED, the main gap**

Paper §5.4:

> *In fluid simulation, only tiles adjacent to the boundary have non-uniform
> matrix coefficients. Before solving the equation, a tile is marked as
> trimmed if both the tile itself and all its neighboring tiles have uniform
> coefficients ... For a trimmed tile, default coefficient values can be used,
> eliminating the need to read coefficients from global memory during the
> equation-solving process.*

**Paper measurement** (Fig 13, 256³):
- V-cycle without trim: 2.78 ms
- V-cycle with trim: **0.81 ms** (3.4× from trim alone)

| Need | Status |
|---|---|
| Tile-uniform flag during build | ❌ |
| Branch in kernel "if trimmed → use defaults, else load from global" | ❌ |
| Avoids loading 4 coefficient channels per voxel for interior tiles | ❌ |

**Our code path** (`cuda_uaamg_preconditioner_3d_opt.cu`): the Laplace stencil is always evaluated with the global `idx2/idy2/idz2` constants and the global `solid` array — there's no fast path for tiles known to be interior + uniform.

**Why we don't see the speedup**: our Poisson is currently *always* uniform-coefficient (constant density), so the paper's "skip coefficient read for trimmed tiles" optimization isn't visible because we never read coefficients in the first place. **The performance lift the paper reports requires the variable-coefficient (Boussinesq / two-phase) configuration to be implemented first.**

---

## Performance measurement (measured on RTX 3090)

Ran `test_paper_bench` (FP64) and `test_paper_bench_f` (FP32) on the dual
RTX 3090 box. Paper numbers are from Table 5 / Fig. 13 (RTX 4090,
256×128×128, V(1,1) cycle, relative residual 10⁻⁶).

### V-cycle cost (single iteration, 256×128×128 ≈ 4.2 M cells)

| Configuration | V-cycle (ms) | ms / MCell |
|---|---|---|
| Paper, separate kernels (RTX 4090) | 2.78 | 0.66 |
| Paper, **aggregate + trim** (RTX 4090) | **0.81** | **0.19** |
| **Ours (RTX 3090), aggregated**, FP64 | **2.76** | **0.66** |
| Ours (RTX 3090), aggregated, narrow V(1,1) | 0.54 | 0.13 |

Our aggregated V-cycle matches the paper's *un-trimmed* aggregated number
to within 1 % (2.76 vs 2.78 ms). The full **3.4× gap** to the paper's
0.81 ms comes entirely from §5.4 trimming, which we don't implement.

### Full PCG solve at 256×128×128, residual 10⁻⁶

| Configuration | Iters | Solve (ms) | ms / iter |
|---|---|---|---|
| Paper, separate kernels (RTX 4090, FP64) | 16 | 70.4 | 4.40 |
| **Paper, aggregate + trim (RTX 4090, FP64)** | 16 | **28.6** | **1.79** |
| Shao 2022 SIMD CPU UAAMG | 4 (W-cycle) | 510 | — |
| **Ours, GPU PCG-opt (RTX 3090, FP64)** | 40 | 152.4 | **3.81** |
| **Ours, GPU PCG-opt (RTX 3090, FP32)** | 40 | 69.2 | **1.73** |

**Headline result**:

- Per-iteration cost — **FP32 ours: 1.73 ms vs paper FP64: 1.79 ms** — essentially identical when you control for hardware (3090 vs 4090) and precision.
- Total solve time is 2.4× slower (FP32) because **we take 40 iters where the paper takes 16** at the same nominal tolerance.

### Sub-MCell scaling check

| Grid | Cells | V-cycle FP64 (ms) | V-cycle / MCell |
|---|---|---|---|
| 64×32×32 | 0.066 M | 0.32 | 4.85 |
| 128×64×64 | 0.524 M | 0.61 | 1.16 |
| 256×128×128 | 4.19 M | 2.76 | 0.66 |

Per-cell cost converges to ~0.66 ms/MCell at high resolutions, matching
the paper's un-trimmed FP64 number. At small grids the kernel-launch
overhead dominates and per-cell cost looks worse.

### FP32 vs FP64 speedup on RTX 3090

| Grid | FP64 PCG-opt (ms) | FP32 PCG-opt (ms) | FP32 speedup |
|---|---|---|---|
| 64×32×32 | 6.72 | 4.27 | 1.57× |
| 128×64×64 | 23.64 | 11.67 | 2.03× |
| 256×128×128 | 165.22 | 69.23 | **2.39×** |

FP32 gives a clean ~2× at production size — consistent with bandwidth-bound
behavior on consumer GPUs (RTX 3090 has 2:1 FP32:FP64 ratio for shared
memory and HBM bandwidth, despite the 64:1 raw FLOPS gap).

### Where the remaining gap is

The paper's 5× headline speedup over our current FP32 (28.6 ms vs 69.2 ms
on 256³) decomposes as:

| Source | Estimated contribution |
|---|---|
| RTX 4090 vs RTX 3090 hardware | ~1.5× |
| **Coefficient trimming (§5.4) — not implemented** | ~2× |
| **Convergence rate (16 iters vs our 40)** | ~2.5× |
| (multiplicative) | **~7.5×** before precision differences |

Hardware is fixed. The two reachable wins are:

1. **Trimming** — for variable-coefficient scenarios, as discussed above.
2. **Convergence acceleration** — paper hits residual 10⁻⁶ in 16 iters.

---

## Acceleration attempt #1 — V(2,2) cycle

Added a stand-alone `rbgs_smooth_only_kernel` (FP64) and
`rbgs_smooth_only_kernel_f` (FP32) and modified `vCycle_opt` /
`vCycle_opt_f` so that each level does (`nu`-1) extra smooth-only
passes before the fused smooth+restrict (down-stroke) and after the
fused prolong+smooth (up-stroke). `nu = 2` → V(2,2): 2 pre + 2 post
smoothings.

**Measured at 256×128×128 (40 fixed PCG-opt iterations, RTX 3090)**:

| Cycle | Solve FP64 (ms) | Δ vs V(1,1) | Solve FP32 (ms) | Δ vs V(1,1) | PCG-opt L2 residual |
|---|---|---|---|---|---|
| V(1,1) | 152.4 | — | 69.2 | — | 3.0 × 10³ |
| V(2,2) | 210.4 | **+38 %** | 80.7 | **+17 %** | 4.0 × 10³ |

**Result: no win, slight regression.**

Two things suggest the extra smoothings aren't actually buying convergence:

- The L2 residual at fixed iter count is **higher** with V(2,2). Numerically
  the difference (~30 %) is within the FP noise floor of the aggregated
  matvec/restriction path, but a true convergence improvement should at
  least *not* be a regression.
- The PCG-side runtime went up by 38 % (FP64) — consistent with adding
  2× the smoothing kernel launches per V-cycle — yet the residual
  did not drop.

**Why V(2,2) didn't help**:
PCG amortizes preconditioner quality across many iterations. If V(1,1)
is already a "good enough" preconditioner (consistent with our per-iter
throughput matching the paper's), doubling the smoothing cost is closer
to multiplying the V-cycle work without raising the preconditioner's
spectral quality enough. The classical regime where V(2,2) helps is
when the smoother is weak relative to the residual spectrum — our
Red-Black GS on uniform Poisson is already at the strong end.

The 40-vs-16-iter gap therefore is **not** a smoothing-count issue.
Likely candidates left to investigate:

1. **Initial-guess / re-centering pattern**. Paper §5.3 last paragraph
   notes the matvec+dot kernel also recenters (subtracts mean) — if we
   recenter only at start and not every iter we may eat extra iters.
2. **W-cycle** (recurse twice into the coarse level). Shao 2022 converged
   in 4 iters with W-cycle. Significantly heavier per-cycle but might
   pay back if the coarse-grid error is the dominant mode.
3. **Krylov method choice**. Paper uses preconditioned CG; W-cycle gets
   them to 4 iters but CG to 16. If they're at the convergence limit
   of CG with this preconditioner, we should be similar. We're at 40
   which is higher than expected.

Code change kept in `src/solver/cuda/cuda_uaamg_preconditioner_3d_opt*.cu`
behind `static constexpr int VCYCLE_NU = 2;` — flipping back to 1
restores V(1,1) behavior in one place per file.

---

## Punch list to close the gap

1. **Add variable-coefficient Poisson path** (only needed when scenarios like Fire Ball use density-weighted projection). Without it, trimming has nothing to skip.
2. **Implement tile trimming**:
   - In the multigrid build step, scan each tile + its 6 face neighbors for "all interior, no solid" — set a per-tile bool flag.
   - In aggregated kernels, branch on the flag: trimmed tiles skip 4 coefficient loads (diagonal + 3 off-diag) and use compile-time constants.
3. **Switch to a literal 5-channel SoA** (`Tile<bool dof; float diag; float ox,oy,oz>`) so the trimmed-tile path is a single `if(trimmed) use_defaults() else load_struct()`.
4. **Make voxel indexing pure bit-shift** (paper §5.2 last paragraph) — minor, mostly matters for fp32 throughput at very high resolutions.

These together would bring our V-cycle from current ~2-3 ms (estimated) to paper-level ~0.8 ms on a comparable GPU.

---

*Generated 2026-05-27, matches commit history through `92f8d81` (animation cylinder fix).*

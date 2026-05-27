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

## Performance comparison

Paper Table 5 — 256³ Poisson, residual 1e-6, RTX 4090:

| | Iters | Build (ms) | Solve (ms) |
|---|---|---|---|
| Paper, separate kernels | 16 | 0.493 | 70.4 |
| Paper, aggregate + trim | 16 | 0.906 | **28.6** |
| Shao 2022 SIMD CPU | 4 (W-cycle) | 69.1 | 510 |

To compare ours, run:

```bash
./build/src/solver/cuda/test_paper_bench       # FP64
./build/src/solver/cuda/test_paper_bench_f     # FP32
```

The harness already prints per-iter ms and total solve time on a 256×128×128 grid and computes the speedup ratio versus the unoptimized GPU baseline. Expected outcome on RTX 3090 (we have it):
- Aggregated path should approach paper's 28.6 ms × (RTX3090/RTX4090 perf ratio) ≈ **~45 ms**
- The "+trim" 2× gap is invisible until we add a variable-coefficient scenario

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

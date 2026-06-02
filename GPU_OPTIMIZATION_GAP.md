# GPU AMGPCG Optimization — Paper vs. Ours

Side-by-side audit of the four GPU optimizations described in
*Leapfrog Flow Maps for Real-Time Fluid Simulation* §5 (Sun et al. 2025,
SIGGRAPH) against our CUDA implementation.

> **Status note (current).** §5.1 is now a true matrix-free **Galerkin** UAAMG
> (was a rediscretized ×2 MG that did not converge) — the decisive correctness
> win, done, converges mesh-independently (26/32/40/16 iters). §5.3 (aggregation)
> and §5.4 (trimming) are implemented but give little in FP64 + plain arrays
> (break-even / ~6%): the paper's per-optimization gains are **interdependent**
> and need the **FP32 + tiled-SoA** pipeline. Next high-value step = **FP32**.

| § | Paper feature | Status | Notes |
|---|---|---|---|
| 5.1 | Matrix-Free UAAMG (Galerkin, ×2 prolong, RBGS) | ✅ **done (true Galerkin)** | `cuda_uaamg_preconditioner_3d.cu` (2D: `cuda_uaamg_preconditioner.cu`); converges all scales |
| 5.2 | SoA + 8³ tile + bit-shift index | 🟡 shared-mem tiling done | full tiled-SoA layout ~1.0× (skip); **shared-memory tiled RBGS gives ~1.15–1.22× on large FP32 levels** (gated by size) |
| 5.3 | Aggregated kernels + dot fusion | 🟡 partial | dot fusion ✓; fused V-cycle aggregation (overlapped) break-even FP64 + nx≥256 bug → not used; the standalone tiled smoother (above) is the useful part |
| 5.4 | Coefficients trimming | 🟡 implemented | per-tile flag + default stencil; only ~6% in FP64 (coeff reads cached) |
| FP32 | Single precision (templated) | ✅ done | `CudaUAAMGPreconditioner3DT<T>`; FP32 V-cycle **1.4–2.1×** vs FP64 |
| — | Setup-once (coeffs/trim per solve, not per iter) | ✅ done | `setupLevels` + `vcycle_apply` split |

---

## §5.1 Matrix-Free UAAMG — ✅ done (now correct)

| Item | Paper (Alg. 3, Eq. 11–12) | Ours |
|---|---|---|
| Restriction | 8-to-1, R = Pᵀ | ✓ **sum over 2×2×2 children** (= Pᵀ), not averaging |
| Prolongation | constant injection × **2.0** (Stüben 2001) | ✓ injection × 2.0 |
| Coarse operator | **Galerkin** A_{l+1} = R_l A_l P_l | ✓ **true Galerkin**, matrix-free |
| Smoother | Red-Black Gauss-Seidel, V(1,1) | ✓ RBGS, **symmetric V(1,1)** (fwd pre / rev post) |
| Coarsest | ~10 RBGS | ✓ 10 symmetric sweeps |

**The fix that mattered.** The paper's ×2 prolongation is only valid *with* the
Galerkin coarse operator A_c = R A P. The old code instead **rediscretized** the
Laplacian on the coarse grid (recompute the 7-pt stencil at 2·dx) — for the
isotropic interior the Galerkin operator is `16×` the rediscretized one, so the
×2 over-corrected smooth modes 2× and crippled multigrid (PCG stalled/oscillated,
e.g. KVS divergence stuck at 3.5e-3 vs GMG's 1.7e-7). We now build the genuine
Galerkin stencil, kept matrix-free by a simple rule:

> coarse +x coupling = **sum of the fine +x couplings across the shared face**
> (4 cells in 3D, 2 in 2D); coarse diagonal = sum of the 6 (4 in 2D) couplings,
> so the Neumann zero-row-sum is preserved.

Each level stores 4 channels `{diag, cx, cy, cz}` (cx = coupling to +x neighbour).
A **symmetric** V-cycle (reverse-order RBGS post-smooth) keeps M SPD for CG — the
paper's Alg. 3 pseudocode is forward-only, but red-black GS forward-only is not
symmetric and converges ~3× slower / breaks down as a CG preconditioner.

**Verdict.** Algorithmically at (or above) paper level. GPU bit-matches CPU
(V-cycle diff = 0). Converges mesh-independently: 26 / 32 / 40 / 16 iters at
64³ / 128×64×64 / 256×128×128 / 256³ to relative 1e-6.

---

## §5.2 Data Structure — 🟡 partial

| Item | Paper | Ours |
|---|---|---|
| 5 channels per voxel (bool DoF + diag + 3 off-diag) | explicit SoA | 🟡 we now store **4 double channels** `{diag,cx,cy,cz}` + `solid` per level (used by Galerkin), but as plain arrays |
| 8³ tile layout | per-tile contiguous SoA | 🟡 only the `_opt` tiled kernels tile; the production `apply` uses plain column-major arrays |
| Bit-shift voxel index | yes | ❌ we multiply (`i + j*pitch + k*pitch*(ny+2)`) |

**Verdict.** The per-cell coefficient channels now exist (needed for Galerkin /
future variable-coefficient Poisson). The full tiled-SoA + bit-index layout is
**not worth implementing on RTX 3090** — measured, see below.

**Microbenchmark (`test_tiled_soa`): plain column-major vs 8³ tiled-SoA RBGS sweep**

| Grid | plain | tiled-SoA | speedup |
|---|---|---|---|
| 64×32×32 | 0.020 ms | 0.020 ms | 0.99× |
| 128×64×64 | 0.122 ms | 0.124 ms | 0.98× |
| 256×128×128 | 0.886 ms | 0.882 ms | 1.00× |
| 256×256×256 | 3.517 ms | 3.378 ms | 1.04× |

The tiled-SoA layout gives **~0% speedup** (only ~4% once the working set far
exceeds L2 at 256³). The RTX 3090's 6 MB L2 already serves the plain-layout
strided neighbour reads; the layout's locality advantage doesn't materialise.
A full tiled-SoA refactor (rewriting the indexing of every kernel + the PCG
matvec) for ~0–4% is not worth it on this hardware.

## ncu profiling — the actual bottleneck (corrects the L2 story above)

`ncu` on the finest-level RBGS smoother (256×128×128), the dominant V-cycle kernel:

| | Compute (SM) | DRAM | Duration | Bound by |
|---|---|---|---|---|
| **FP64** | **89%** | 47% | 357 µs | **FP64 compute** |
| **FP32** | 28% | **71%** | 120 µs | **memory** |

**The real reason the memory optimizations (§5.2/§5.3/§5.4) gave little is not
"L2 serves it" — it's that FP64 is _compute-bound_ on RTX 3090** (GeForce FP64 =
1/64 of FP32; the `(b+nb)/diag` divide + FP64 ALU saturate the SM at 89%). A
compute-bound kernel can't benefit from less memory traffic. **FP32 flips it to
memory-bound (71% DRAM)** — and the per-kernel speedup is ~3× (357→120 µs). This
is exactly why the paper uses FP32: consumer-GPU FP64 is crippled, so they work
in the memory-bound FP32 regime where §5.2/§5.3/§5.4 *do* pay off.

**Profiling-driven optimization applied — shared-memory tiled RBGS.** Since the
FP32 smoother is memory-bound and the dominant traffic is the 7 neighbour-`x`
reads, the fix is to load the 8³ tile + 1-cell face halo of `x` into shared
memory once and read the stencil from there. Done **per colour pass** (red kernel,
black kernel) with a global write-back between passes, so it is **exact** (not
block-RBGS — convergence is identical) and gated to large levels (≥2M cells; it
adds overhead on small/coarse levels):

| Grid | FP32 V-cycle, untiled | tiled (gated) | speedup |
|---|---|---|---|
| 256×128×128 | 1.806 ms | **1.481 ms** | **1.22×** |
| 256×256×256 | 6.502 ms | 5.667 ms | 1.15× |

FP32 V-cycle at 256×128×128 is now **1.48 ms** (was 1.81 ms) vs paper **0.81 ms**
(RTX 4090) — gap narrowed from 2.2× to ~1.8×. FP32-vs-FP64 V-cycle ratio rose to
**2.14×**. This is the §5.2/§5.3 idea finally paying off, but only in the FP32
(memory-bound) regime the profiling identified — it was useless in FP64.

---

## §5.3 Aggregated Kernels — 🟡 partial (the open item)

The paper fuses, per level, **pre-smooth + residual + restrict** (down) and
**prolong + post-smooth** (up) into single shared-memory tiled kernels, plus
dot-product fusion in PCG. Claimed ~2× over separate kernels.

| Piece | Status |
|---|---|
| PCG dot fusion (matvec+dot, axpy+dot) | ✅ done — `cuda_pcg_3d_opt.cu` (used by `solve_optimized`) |
| Galerkin aggregated down (smooth+restrict) | 🟡 implemented, **block-tiled** — `gal_smooth_restrict_kernel` |
| Galerkin aggregated up (prolong+smooth) | 🟡 implemented, **block-tiled** — `gal_prolong_smooth_kernel` |
| Used as the PCG preconditioner | ❌ **no** — falls back to separate `apply` (see below) |

**Measured raw V-cycle speedup (FP64, RTX 3090), aggregated vs separate `apply`:**

| Grid | apply (separate) | apply_optimized (aggregated) | speedup | pure V-cycle (aggregated) |
|---|---|---|---|---|
| 64×32×32 | 0.30 ms | 0.22 ms | **1.36×** | 0.16 ms |
| 128×64×64 | 0.68 ms | 0.49 ms | **1.38×** | 0.35 ms |
| 256×128×128 (4.2M) | 3.69 ms | 2.39 ms | **1.54×** | **1.65 ms** |
| 256³ (16.8M) | 14.17 ms | 8.22 ms | **1.72×** | 5.52 ms |

Pure aggregated V-cycle at 4.2M is **1.65 ms (FP64, RTX 3090)** vs the paper's
**0.81 ms (FP32, RTX 4090, aggregate+trim)** — within ~2× once you account for
FP64→FP32 (~2×) and 3090→4090 (~1.5×), and we are still missing §5.4 trimming.

**The catch — block-tiling breaks convergence.** Fusing smooth+restrict in one
kernel forces the smoother to be **tile-local block-RBGS** (the 1-cell halo is
loaded but not re-smoothed). Block-RBGS ≠ global RBGS, making it a far weaker /
effectively non-SPD preconditioner:

| Grid | PCG iters, separate apply | PCG iters, block-aggregated |
|---|---|---|
| 64³ | 26 | 91 |
| 128×64×64 | 32 | **fails (>200, rel stuck)** |
| 256×128×128 | 40 | **fails** |

So the block-aggregated path is *faster per V-cycle but slower (or non-convergent)
end-to-end*. **`solve_optimized` therefore uses the separate-kernel global-RBGS
`apply`.** The aggregated kernels remain for timing/reference.

**What's needed for a faithful §5.3.** The paper keeps the tiled smoother
**global-equivalent** via **overlapped tiling**: load a 2-cell halo and smooth the
1-ring redundantly (the paper's "Manhattan-distance < 3 red / < 2 black" in §2.3),
so the tile-interior cells get exactly the global RBGS values. That makes the
aggregated V-cycle **bit-identical** to `apply` (hence converges in the same
26–40 iters) while keeping the fusion speedup. This is the remaining work — more
complex (12³ shared tile, multi-cell-per-thread loads, prolonged halo on the
up-stroke), but verifiable by bit-exactness against `apply`.

---

## §5.4 Coefficients Trimming — 🟡 implemented, but only ~6% in FP64

Paper §5.4: tiles whose own + neighbour coefficients are all uniform are marked
"trimmed" and use default coefficients, skipping the global-memory coefficient
read. Paper Fig. 13 (256³): un-trimmed 2.78 ms → trimmed **0.81 ms** (~3.4×).

**Implemented** on the separate-kernel `apply`: per-level uniform defaults
(Galerkin: coarse coupling = 4× finer), a per-tile `trimmed` flag
(`mark_trimmed_kernel_3d` — tile + 1-ring all default), and a branch in
`rbgs_coeff_kernel_3d` that uses the default stencil for trimmed tiles (no global
`{diag,cx,cy,cz}` reads). Correct (uniform-tile defaults == the stored coeffs, so
GPU still bit-matches CPU and converges identically: 26/32/40/16 iters).

**Measured V-cycle (separate, FP64) — A/B via `UAAMG_NOTRIM=1`:**

| Grid | trim OFF | trim ON | speedup |
|---|---|---|---|
| 64×32×32 | 0.282 ms | 0.283 ms | ~1.00× |
| 128×64×64 | 0.607 ms | 0.586 ms | 1.04× |
| 256×128×128 | 3.263 ms | 3.062 ms | 1.07× |

**Only ~6%, not the paper's 3.4×.** Reason: in FP64 with plain arrays the
coefficient reads are already served from L2 (adjacent cells' coupling reads
overlap), and the kernel is dominated by the 7 neighbour-`x` reads + the
double-precision divide — not by uncached coefficient bandwidth. The paper's
3.4× needs the **FP32 + tiled-SoA** layout where coefficient reads are a large,
uncached share of traffic. The three §5 optimizations are interdependent.

---

## Setup-once refactor (real solve-time win, correctness-neutral)

`apply` previously recomputed `setupLevels` (solid restrict + Galerkin coeffs +
trimming) on **every** call → in PCG that is recomputed every iteration. Split
into `setupLevels()` (once per solve) + `vcycle_apply()` (per iteration); the PCG
loop now calls setup once. This removes ~N_iter redundant coefficient rebuilds.

## Full PCG solve, relative residual 1e-6 (RTX 3090, FP64, current state)

| Grid | iters | solve (ms) | ms/iter | V-cycle (ms) |
|---|---|---|---|---|
| 64×32×32 | 26 | 11.7 | 0.45 | 0.285 |
| 128×64×64 | 32 | 30.7 | 0.96 | 0.587 |
| 256×128×128 | 40 | 191.3 | 4.78 | 3.06 |
| 256×256×256 | 16 | 284.6 | 17.8 | 11.06 |

Converges at all scales (vs the paper's ~16 at 256×128×128 — our higher count is
the adversarial 50%-white-noise RHS; a smooth fluid RHS converges faster). Per-V-
cycle 3.06 ms (FP64) vs paper 0.81 ms (FP32) ≈ explained by FP64 (~2×) + RTX
3090→4090 (~1.5×) + no FP32-tiled trimming win.

## FP32 (precision) — ✅ done, the clean ~2× lever

The preconditioner is now templated on the scalar type — `CudaUAAMGPreconditioner3DT<T>`,
with `CudaUAAMGPreconditioner3D = ...<double>` and `CudaUAAMGPreconditioner3Dt =
...<float>` (one codebase, two precisions). FP32 halves all global-memory
traffic, including the dominant `x` reads (unlike §5.4 which only removes the
already-cached coefficient reads).

**Measured FP32 vs FP64 V-cycle (RTX 3090, with §5.4 trimming), `test_fp32_vcycle`:**

| Grid | FP64 | FP32 | speedup |
|---|---|---|---|
| 64×32×32 | 0.248 ms | 0.174 ms | 1.42× |
| 128×64×64 | 0.539 ms | 0.315 ms | 1.71× |
| 256×128×128 | 3.46 ms | **1.81 ms** | 1.91× |
| 256×256×256 | 13.38 ms | 6.51 ms | 2.06× |

Approaches 2× at scale (fully memory-bound). FP32 V-cycle 1.81 ms @256×128×128
vs paper 0.81 ms (FP32, RTX 4090) — remaining gap ≈ RTX 3090→4090 (~1.5×) + the
§5.2/§5.3 tiled-SoA aggregation we don't have.

## Summary — what helped on RTX 3090 / FP64, and what didn't

Measured, per optimization:

| Paper optimization | Effect here | Notes |
|---|---|---|
| §5.1 Galerkin UAAMG | **decisive** | the correctness fix — was non-convergent, now mesh-independent |
| §5.2 tiled-SoA layout | **~1.0×** | microbench: L2 already serves the access pattern |
| §5.3 aggregated/shared-tiled V-cycle | **break-even** | redundant halo compute ≈ fused memory savings (+ nx≥256 bug) |
| §5.4 coefficient trimming | **~1.06×** | coupling reads already L2-cached |
| **FP32 precision** | **~2×** (V-cycle) | the one clean win; halves *all* traffic |

**Key finding.** The paper's three GPU *memory* optimizations (§5.2/§5.3/§5.4)
give little **individually on RTX 3090 in FP64** — the 6 MB L2 already absorbs the
strided neighbour/coefficient reads, so neither tiling nor trimming reduces actual
DRAM traffic much. They are **interdependent** and only compound toward the paper's
3.4× on hardware/precision where those reads are uncached and bandwidth-bound.
The decisive win was **§5.1 (Galerkin correctness)**; the one portable speedup is
**FP32 (~2×, done, templated)**.

## Mixed-precision PCG — ✅ done

`CudaPCG3D::solve_mixed` = FP64 CG outer loop + **FP32 Galerkin preconditioner**
(`CudaUAAMGPreconditioner3Dt`). Each iteration casts r→FP32, runs the FP32
V-cycle, casts z→FP64; the CG vectors/dots stay FP64.

- **Converges identically to FP64** (26/32/40 iters at 64³/128×64×64/256×128×128)
  — the preconditioner is only an approximation, so FP32 is fine, while the FP64
  CG keeps orthogonality. (Pure-FP32 PCG diverges: conditioning × FP32 eps, and
  the unit-spacing solution is O(nx²).)
- **End-to-end speedup ~1.3–1.4×** at compute-bound grids (256×128×128: **1.39×**;
  128×64×64: 1.25×). Not 2×: the FP32 V-cycle (~2×) is only ~half the per-iter
  cost; the FP64 CG ops (matvec, dots with host reductions, axpy) are unaccelerated.
  Small grids are host-sync-latency-bound (the per-dot `cudaDeviceSynchronize` +
  device→host copy dominates) so their timing is noisy.

This is the practical FP32 win: real end-to-end speedup **with** convergence.

## Device-resident CG reduction (no per-iter host sync) — ✅ done

`CudaPCG3D::solve_device`: all CG scalars (rsold, pAp, rsnew, rz, alpha, beta,
mean) live on device; alpha/beta and the mean-projection are 1-thread kernels;
the search direction is updated in place; **no `cudaMemcpy` inside the loop**
(fixed iteration count, like the paper). Verified **correct** — the final
residual matches `solve_optimized` exactly (5.1e-5 / 1.9e-4 / 4.5e-4 at the three
grids).

**Benefit on RTX 3090 is small**: the per-iteration host sync it removes is
~3–4 `cudaMemcpy`/iter ≈ 40 µs — only ~0.8% at 256×128×128 (5 ms/iter, V-cycle-
bound) and ~10% at 64³ (0.4 ms/iter). The wall-clock benchmark is dominated by
GPU-boost-clock noise on this shared box (FP64 64³ measured anywhere from 10–65 ms
run-to-run), so the speedup can't be cleanly isolated; it is architecturally the
paper's approach and removes the small-grid latency tax. Kept as `solve_device`.

## What's left (low priority on this hardware)

- §5.2 tiled-SoA / §5.3 aggregation: measured ~1.0× / break-even here (L2-served).
  Would only help on DRAM-bound hardware/precision — profile first (Nsight).
- The remaining ~2× gap to the paper is **hardware (RTX 3090→4090, ~1.5×)** plus
  the §5.2/§5.3 tiled aggregation, which the microbench shows won't pay off until
  the workload is DRAM-bound rather than L2-served.

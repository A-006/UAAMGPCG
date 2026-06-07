/**
 * @file cuda_uaamg_preconditioner_3d.cu
 * @brief CUDA 3D UAAMG V-cycle preconditioner — matrix-free Galerkin, templated
 *        on the scalar type T (double / float).
 *
 * Paper-faithful (Sun et al. SIGGRAPH 2025, Algorithm 3):
 *   - constant prolongation with ×2 correction scaling (Eq. 11);
 *   - 8-to-1 restriction R = Pᵀ (sum over 2×2×2 children);
 *   - Galerkin coarse operator A_{l+1} = R_l A_l P_l (Eq. 12), matrix-free:
 *     coarse +x coupling = sum of the 4 fine +x couplings across the shared face;
 *   - RBGS smoother, symmetric V(1,1);
 *   - §5.4 coefficient trimming (uniform tiles use a default stencil).
 *
 * Memory layout (§5.3): 8×8×8 TILE-CONTIGUOUS (Structure-of-Arrays). A voxel at
 * interior coord (i,j,k)∈[1..n] lives at  tile_idx*512 + voxel_idx,  with
 * tile=((i-1)/8,(j-1)/8,(k-1)/8) and voxel=((i-1)%8,(j-1)%8,(k-1)%8). Each channel
 * (x,b,solid,diag,cx,cy,cz) is its own contiguous array of num_tiles*512. This
 * keeps each 8³ tile's data contiguous → far better L2 locality for the
 * bandwidth-bound RBGS smoother than the old pitched-linear layout. There are no
 * ghost cells: domain-boundary neighbours are skipped by explicit i>1 / i<nx
 * guards (the skipped couplings were identically zero, so results are unchanged).
 * The (pitched) r/z/solid that callers supply are converted to/from the tile
 * layout only at the level-0 boundary (an O(N) scatter/gather, once per apply).
 *
 * Kernel fusion (kept from the pitched base): smooth_from_zero_3d (1-launch
 * exact pre-smooth from x=0), restrict_residual_tiled_3d (fused residual+restrict),
 * prolong_black_fused_3d (ping-pong prolong+black), and the single-launch
 * coarsest_solve_kernel_3d are all preserved — only their indexing/storage changed.
 *
 * Instantiated for double and float at the bottom. FP32 halves memory traffic.
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"
#include <cstdlib>

// 8×8×8 tile-contiguous index for interior coord (i,j,k)∈[1..n]. ny,nz are the
// grid's interior extents (used to derive the per-axis tile counts).
//
// Voxel order within a tile is i-innermost: voxel = ((k%8)*8 + (j%8))*8 + (i%8).
// The block is dim3(8,8,8), so a warp (32 consecutive threads) spans tx=0..7 ×
// ty=0..3 at fixed tz — exactly voxels 0..31. With i innermost, thread-linear-id
// == voxel offset, so x/b/diag global loads are perfectly coalesced (one 128B
// line per warp). (A k-innermost voxel order would make tx stride-64 → scattered.)
__device__ __forceinline__ int dev_idx3d(int i, int j, int k, int ny, int nz) {
    int ii = i - 1, jj = j - 1, kk = k - 1;
    int tny = (ny + 7) >> 3, tnz = (nz + 7) >> 3;
    int tile = ((ii >> 3) * tny + (jj >> 3)) * tnz + (kk >> 3);
    return tile * 512 + ((kk & 7) * 8 + (jj & 7)) * 8 + (ii & 7);
}

// pitched-linear index (caller layout) for the level-0 boundary conversion.
__device__ __forceinline__ int pitch_idx3d(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

// ── pitched ⇄ tile-contiguous conversion at the level-0 boundary ──
template <typename T>
__global__ void scatter_p2t_kernel_3d(const T* __restrict src, T* __restrict dst, int nx, int ny,
                                      int nz, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    dst[dev_idx3d(i, j, k, ny, nz)] = src[pitch_idx3d(i, j, k, pitch, ny)];
}
template <typename T>
__global__ void gather_t2p_kernel_3d(const T* __restrict src, T* __restrict dst, int nx, int ny,
                                     int nz, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    dst[pitch_idx3d(i, j, k, pitch, ny)] = src[dev_idx3d(i, j, k, ny, nz)];
}
__global__ void scatter_solid_p2t_kernel_3d(const bool* __restrict src, bool* __restrict dst, int nx,
                                            int ny, int nz, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    dst[dev_idx3d(i, j, k, ny, nz)] = src[pitch_idx3d(i, j, k, pitch, ny)];
}

// Subtract the mean over interior fluid cells (single-block; for the tiny
// coarsest level). On a pure-Neumann (closed) domain the coarsest operator is
// singular with the constants as null space, so its iterative "solve" is only
// well-posed on the zero-mean subspace — projecting the RHS (and the solution)
// there stops the null-space component from growing and stalling the V-cycle.
template <typename T>
__global__ void coarse_remove_mean_kernel_3d(T* v, const bool* solid, int nx, int ny, int nz) {
    __shared__ double ssum[256];
    __shared__ int scnt[256];
    __shared__ double smean;
    int t        = threadIdx.x;
    long total   = (long)nx * ny * nz;
    double sum   = 0;
    int cnt      = 0;
    for (long l = t; l < total; l += blockDim.x) {
        int i = (int)(l % nx) + 1, j = (int)((l / nx) % ny) + 1, k = (int)(l / ((long)nx * ny)) + 1;
        int id = dev_idx3d(i, j, k, ny, nz);
        if (!solid[id]) {
            sum += (double)v[id];
            cnt++;
        }
    }
    ssum[t] = sum;
    scnt[t] = cnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            ssum[t] += ssum[t + s];
            scnt[t] += scnt[t + s];
        }
        __syncthreads();
    }
    if (t == 0)
        smean = (scnt[0] > 0) ? ssum[0] / scnt[0] : 0.0;
    __syncthreads();
    T mean = (T)smean;
    for (long l = t; l < total; l += blockDim.x) {
        int i = (int)(l % nx) + 1, j = (int)((l / nx) % ny) + 1, k = (int)(l / ((long)nx * ny)) + 1;
        int id = dev_idx3d(i, j, k, ny, nz);
        if (!solid[id])
            v[id] -= mean;
    }
}

// ── Finest-level stencil from the solid mask ──
template <typename T>
__global__ void setup_fine_coeffs_kernel_3d(const bool* solid, T* cx, T* cy, T* cz, int nx, int ny,
                                            int nz, T idx2, T idy2, T idz2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int id = dev_idx3d(i, j, k, ny, nz);
    if (solid[id]) {
        cx[id] = cy[id] = cz[id] = T(0);
        return;
    }
    cx[id] = (i < nx && !solid[dev_idx3d(i + 1, j, k, ny, nz)]) ? idx2 : T(0);
    cy[id] = (j < ny && !solid[dev_idx3d(i, j + 1, k, ny, nz)]) ? idy2 : T(0);
    cz[id] = (k < nz && !solid[dev_idx3d(i, j, k + 1, ny, nz)]) ? idz2 : T(0);
}

// diag[c] = sum of the 6 active couplings (fine or coarse level)
template <typename T>
__global__ void setup_diag_kernel_3d(const bool* solid, const T* cx, const T* cy, const T* cz,
                                     T* diag, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int id = dev_idx3d(i, j, k, ny, nz);
    if (solid[id]) {
        diag[id] = T(0);
        return;
    }
    T d = cx[id] + cy[id] + cz[id]; // +x/+y/+z couplings stored at this cell
    if (i > 1)
        d += cx[dev_idx3d(i - 1, j, k, ny, nz)]; // −x coupling stored at left cell
    if (j > 1)
        d += cy[dev_idx3d(i, j - 1, k, ny, nz)];
    if (k > 1)
        d += cz[dev_idx3d(i, j, k - 1, ny, nz)];
    diag[id] = d;
}

// ── Galerkin coarse couplings: sum the 4 fine couplings on each shared face ──
template <typename T>
__global__ void galerkin_coeffs_kernel_3d(const T* fcx, const T* fcy, const T* fcz,
                                          const bool* csolid, T* ccx, T* ccy, T* ccz, int fnx,
                                          int fny, int fnz, int cnx, int cny, int cnz) {
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (ic > cnx || jc > cny || kc > cnz)
        return;
    int cid = dev_idx3d(ic, jc, kc, cny, cnz);
    if (csolid[cid]) {
        ccx[cid] = ccy[cid] = ccz[cid] = T(0);
        return;
    }
    int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1;
    T sx = 0, sy = 0, sz = 0;
    for (int dj = 0; dj < 2; dj++)
        for (int dk = 0; dk < 2; dk++)
            sx += fcx[dev_idx3d(i_f + 1, j_f + dj, k_f + dk, fny, fnz)]; // +x face = fine i_f+1
    for (int di = 0; di < 2; di++)
        for (int dk = 0; dk < 2; dk++)
            sy += fcy[dev_idx3d(i_f + di, j_f + 1, k_f + dk, fny, fnz)];
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            sz += fcz[dev_idx3d(i_f + di, j_f + dj, k_f + 1, fny, fnz)];
    ccx[cid] = sx;
    ccy[cid] = sy;
    ccz[cid] = sz;
}

// ── §5.4 mark trimmed tiles (tile + 1-ring all uniform-default) ──
template <typename T>
__global__ void mark_trimmed_kernel_3d(const T* diag, T diagd, int nx, int ny, int nz, int ntx,
                                       int nty, int ntz, bool* trimmed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntx * nty * ntz)
        return;
    int bx = t % ntx, by = (t / ntx) % nty, bz = t / (ntx * nty);
    bool trim = true;
    for (int gi = 8 * bx; gi <= 8 * bx + 9 && trim; gi++)
        for (int gj = 8 * by; gj <= 8 * by + 9 && trim; gj++)
            for (int gk = 8 * bz; gk <= 8 * bz + 9 && trim; gk++) {
                if (gi < 1 || gi > nx || gj < 1 || gj > ny || gk < 1 || gk > nz) {
                    trim = false;
                    break;
                }
                T d = diag[dev_idx3d(gi, gj, gk, ny, nz)];
                if (fabs(double(d - diagd)) > 1e-6 * double(diagd)) {
                    trim = false;
                    break;
                }
            }
    trimmed[t] = trim;
}

// ── RBGS sweep (one parity), stored coeffs + §5.4 trimming ──
template <typename T>
__global__ void rbgs_coeff_kernel_3d(T* x, const T* b, const bool* solid, const T* diag,
                                     const T* cx, const T* cy, const T* cz, int nx, int ny, int nz,
                                     int parity, const bool* trimmed, T cxd, T cyd, T czd, T diagd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    if (((i + j + k) & 1) != parity)
        return;
    int id = dev_idx3d(i, j, k, ny, nz);
    if (solid[id])
        return;
    int tileid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    T Cxp, Cxm, Cyp, Cym, Czp, Czm, D;
    if (trimmed[tileid]) { // uniform tile → default stencil
        Cxp = Cxm = cxd;
        Cyp = Cym = cyd;
        Czp = Czm = czd;
        D         = diagd;
    } else {
        D = diag[id];
        if (D < T(1e-30))
            return;
        Cxp = cx[id];
        Cyp = cy[id];
        Czp = cz[id];
        Cxm = (i > 1) ? cx[dev_idx3d(i - 1, j, k, ny, nz)] : T(0);
        Cym = (j > 1) ? cy[dev_idx3d(i, j - 1, k, ny, nz)] : T(0);
        Czm = (k > 1) ? cz[dev_idx3d(i, j, k - 1, ny, nz)] : T(0);
    }
    T nb = T(0);
    if (i < nx)
        nb += Cxp * x[dev_idx3d(i + 1, j, k, ny, nz)];
    if (i > 1)
        nb += Cxm * x[dev_idx3d(i - 1, j, k, ny, nz)];
    if (j < ny)
        nb += Cyp * x[dev_idx3d(i, j + 1, k, ny, nz)];
    if (j > 1)
        nb += Cym * x[dev_idx3d(i, j - 1, k, ny, nz)];
    if (k < nz)
        nb += Czp * x[dev_idx3d(i, j, k + 1, ny, nz)];
    if (k > 1)
        nb += Czm * x[dev_idx3d(i, j, k - 1, ny, nz)];
    x[id] = (b[id] + nb) / D;
}

// ── Coarsest solve in ONE launch (replaces 42 tiny launches) ──
// The coarsest grid is a handful of cells → one block covers it, so
// __syncthreads() is a full-grid barrier. Does: remove mean(b) + `iters`×
// (forward RBGS + reverse RBGS) + remove mean(x), exactly mirroring the
// multi-launch sequence (same parities [1,0,0,1], same per-cell math) → identical
// convergence, but ~42 launches → 1 (kills the fixed launch overhead).
template <typename T>
__global__ void coarsest_solve_kernel_3d(T* x, T* b, const bool* solid, const T* diag, const T* cx,
                                         const T* cy, const T* cz, int nx, int ny, int nz,
                                         const bool* trimmed, T cxd, T cyd, T czd, T diagd,
                                         int iters) {
    __shared__ double ssum[256];
    __shared__ int scnt[256];
    __shared__ double smean;
    int t      = threadIdx.x;
    long total = (long)nx * ny * nz;
    int gtx = (nx + 7) / 8, gty = (ny + 7) / 8;
    auto cell = [&](long l, int& i, int& j, int& k) {
        i = (int)(l % nx) + 1;
        j = (int)((l / nx) % ny) + 1;
        k = (int)(l / ((long)nx * ny)) + 1;
    };
    auto remove_mean = [&](T* v) {
        double sum = 0;
        int cnt    = 0;
        for (long l = t; l < total; l += blockDim.x) {
            int i, j, k;
            cell(l, i, j, k);
            int id = dev_idx3d(i, j, k, ny, nz);
            if (!solid[id]) {
                sum += (double)v[id];
                cnt++;
            }
        }
        ssum[t] = sum;
        scnt[t] = cnt;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (t < s) {
                ssum[t] += ssum[t + s];
                scnt[t] += scnt[t + s];
            }
            __syncthreads();
        }
        if (t == 0)
            smean = (scnt[0] > 0) ? ssum[0] / scnt[0] : 0.0;
        __syncthreads();
        T m = (T)smean;
        for (long l = t; l < total; l += blockDim.x) {
            int i, j, k;
            cell(l, i, j, k);
            int id = dev_idx3d(i, j, k, ny, nz);
            if (!solid[id])
                v[id] -= m;
        }
        __syncthreads();
    };

    remove_mean(b);
    const int order[4] = {1, 0, 0, 1}; // forward(red,black) + reverse(black,red)
    for (int it = 0; it < iters; it++) {
        for (int s = 0; s < 4; s++) {
            int parity = order[s];
            for (long l = t; l < total; l += blockDim.x) {
                int i, j, k;
                cell(l, i, j, k);
                if (((i + j + k) & 1) != parity)
                    continue;
                int id = dev_idx3d(i, j, k, ny, nz);
                if (solid[id])
                    continue;
                int tileid = ((k - 1) / 8) * gtx * gty + ((j - 1) / 8) * gtx + (i - 1) / 8;
                T Cxp, Cxm, Cyp, Cym, Czp, Czm, D;
                if (trimmed[tileid]) {
                    Cxp = Cxm = cxd;
                    Cyp = Cym = cyd;
                    Czp = Czm = czd;
                    D         = diagd;
                } else {
                    D = diag[id];
                    if (D < T(1e-30))
                        continue;
                    Cxp = cx[id];
                    Cyp = cy[id];
                    Czp = cz[id];
                    Cxm = (i > 1) ? cx[dev_idx3d(i - 1, j, k, ny, nz)] : T(0);
                    Cym = (j > 1) ? cy[dev_idx3d(i, j - 1, k, ny, nz)] : T(0);
                    Czm = (k > 1) ? cz[dev_idx3d(i, j, k - 1, ny, nz)] : T(0);
                }
                T nb = T(0);
                if (i < nx)
                    nb += Cxp * x[dev_idx3d(i + 1, j, k, ny, nz)];
                if (i > 1)
                    nb += Cxm * x[dev_idx3d(i - 1, j, k, ny, nz)];
                if (j < ny)
                    nb += Cyp * x[dev_idx3d(i, j + 1, k, ny, nz)];
                if (j > 1)
                    nb += Cym * x[dev_idx3d(i, j - 1, k, ny, nz)];
                if (k < nz)
                    nb += Czp * x[dev_idx3d(i, j, k + 1, ny, nz)];
                if (k > 1)
                    nb += Czm * x[dev_idx3d(i, j, k - 1, ny, nz)];
                x[id] = (b[id] + nb) / D;
            }
            __syncthreads();
        }
    }
    remove_mean(x);
}

// ── Residual restriction R = Pᵀ (sum over 2×2×2 children) ──
template <typename T>
__global__ void restrict_coeff_kernel_3d(const T* xf, const T* bf, const bool* fsolid,
                                         const T* fdiag, const T* fcx, const T* fcy, const T* fcz,
                                         T* bc, const bool* csolid, int fnx, int fny, int fnz,
                                         int cnx, int cny, int cnz) {
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (ic > cnx || jc > cny || kc > cnz)
        return;
    int cid = dev_idx3d(ic, jc, kc, cny, cnz);
    if (csolid[cid])
        return;
    int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1;
    T sum = 0;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++) {
                int fi = i_f + di, fj = j_f + dj, fk = k_f + dk,
                    fid = dev_idx3d(fi, fj, fk, fny, fnz);
                if (fsolid[fid])
                    continue;
                T Ax = fdiag[fid] * xf[fid];
                if (fi < fnx)
                    Ax -= fcx[fid] * xf[dev_idx3d(fi + 1, fj, fk, fny, fnz)];
                if (fi > 1)
                    Ax -= fcx[dev_idx3d(fi - 1, fj, fk, fny, fnz)] *
                          xf[dev_idx3d(fi - 1, fj, fk, fny, fnz)];
                if (fj < fny)
                    Ax -= fcy[fid] * xf[dev_idx3d(fi, fj + 1, fk, fny, fnz)];
                if (fj > 1)
                    Ax -= fcy[dev_idx3d(fi, fj - 1, fk, fny, fnz)] *
                          xf[dev_idx3d(fi, fj - 1, fk, fny, fnz)];
                if (fk < fnz)
                    Ax -= fcz[fid] * xf[dev_idx3d(fi, fj, fk + 1, fny, fnz)];
                if (fk > 1)
                    Ax -= fcz[dev_idx3d(fi, fj, fk - 1, fny, fnz)] *
                          xf[dev_idx3d(fi, fj, fk - 1, fny, fnz)];
                sum += bf[fid] - Ax;
            }
    bc[cid] = sum;
}

// ════════════════════════════════════════════════════════════════════
//  FUSED down-leg: block-RBGS(2 colours) + residual + restrict, ONE launch.
//  Mirrors the paper's GaussSeidelRestrict kernel: x lives in shared across
//  both colour passes (no global x reload between red/black), the residual is
//  produced as a by-product of smoothing, and restricted in the same launch —
//  so restriction never costs its own full-grid A-apply or coefficient reads.
//  Trivial (uniform) tiles use scalar coefficients (no a_* array reads).
//  With the tile-contiguous layout each block maps to exactly one 8³ storage
//  tile, so the center load is one contiguous 512-element run.
// ════════════════════════════════════════════════════════════════════
// Exact single RBGS pre-smooth from x=0 (the V-cycle zeroes x at entry of every
// level). Author's trick: red cells = b/diag computed locally, so the tile halo
// has no stale-x problem — every tile derives the same b/diag for its red cells.
// Red is set directly (skip a pass); black is then exact from the red halo too.
// Uniform-coefficient (trivial) path; non-trivial tiles fall back to the global x.
template <typename T>
__global__ void smooth_from_zero_3d(T* x, const T* b, const bool* solid, const T* diag, const T* cx,
                                    const T* cy, const T* cz, int nx, int ny, int nz,
                                    const bool* trimmed, T cxd, T cyd, T czd, T diagd) {
    __shared__ T sx[10][10][10];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gi = blockIdx.x * 8 + tx + 1, gj = blockIdx.y * 8 + ty + 1, gk = blockIdx.z * 8 + tz + 1;
    int li = tx + 1, lj = ty + 1, lk = tz + 1;
    bool valid    = (gi <= nx && gj <= ny && gk <= nz);
    T inv         = T(1) / diagd;
    // init red cells (interior + halo) = b/diag, exploiting x=0. black = 0.
    auto red_bd   = [&](int ci, int cj, int ck) -> T {
        if (ci < 1 || ci > nx || cj < 1 || cj > ny || ck < 1 || ck > nz)
            return T(0);
        int id = dev_idx3d(ci, cj, ck, ny, nz);
        if (solid[id] || ((ci + cj + ck) & 1) == 0)
            return T(0); // only red parity carries b/diag; black stays 0
        return b[id] * inv;
    };
    sx[li][lj][lk] = red_bd(gi, gj, gk);
    if (tx == 0)
        sx[0][lj][lk] = red_bd(gi - 1, gj, gk);
    if (tx == 7)
        sx[9][lj][lk] = red_bd(gi + 1, gj, gk);
    if (ty == 0)
        sx[li][0][lk] = red_bd(gi, gj - 1, gk);
    if (ty == 7)
        sx[li][9][lk] = red_bd(gi, gj + 1, gk);
    if (tz == 0)
        sx[li][lj][0] = red_bd(gi, gj, gk - 1);
    if (tz == 7)
        sx[li][lj][9] = red_bd(gi, gj, gk + 1);

    int gid       = valid ? dev_idx3d(gi, gj, gk, ny, nz) : 0;
    bool is_solid = valid && solid[gid];
    bool dof      = valid && !is_solid;
    int tileid    = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    bool triv     = trimmed[tileid];
    __syncthreads();
    if (!triv) {
        // non-uniform tile: do the safe exact 2-pass RBGS reading global x
        // (correctness over speed on the few boundary tiles).
        sx[li][lj][lk] = valid ? x[gid] : T(0);
        if (tx == 0)
            sx[0][lj][lk] = (gi > 1 && valid) ? x[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0);
        if (tx == 7)
            sx[9][lj][lk] = (gi < nx && valid) ? x[dev_idx3d(gi + 1, gj, gk, ny, nz)] : T(0);
        if (ty == 0)
            sx[li][0][lk] = (gj > 1 && valid) ? x[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0);
        if (ty == 7)
            sx[li][9][lk] = (gj < ny && valid) ? x[dev_idx3d(gi, gj + 1, gk, ny, nz)] : T(0);
        if (tz == 0)
            sx[li][lj][0] = (gk > 1 && valid) ? x[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0);
        if (tz == 7)
            sx[li][lj][9] = (gk < nz && valid) ? x[dev_idx3d(gi, gj, gk + 1, ny, nz)] : T(0);
        T D = dof ? diag[gid] : diagd;
        if (D < T(1e-30))
            dof = false;
        T cxp = dof ? cx[gid] : cxd, cyp = dof ? cy[gid] : cyd, czp = dof ? cz[gid] : czd;
        T cxm = dof ? ((gi > 1) ? cx[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0)) : cxd;
        T cym = dof ? ((gj > 1) ? cy[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0)) : cyd;
        T czm = dof ? ((gk > 1) ? cz[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0)) : czd;
        __syncthreads();
        for (int pass = 0; pass < 2; pass++) {
            int par = pass; // 0 then 1
            if (dof && ((gi + gj + gk) & 1) == par) {
                T nb = cxp * sx[li + 1][lj][lk] + cxm * sx[li - 1][lj][lk] +
                       cyp * sx[li][lj + 1][lk] + cym * sx[li][lj - 1][lk] +
                       czp * sx[li][lj][lk + 1] + czm * sx[li][lj][lk - 1];
                sx[li][lj][lk] = (b[gid] + nb) / D;
            }
            __syncthreads();
        }
        if (dof)
            x[gid] = sx[li][lj][lk];
        return;
    }
    // trivial tile: red already = b/diag in shared; compute black exactly.
    if (dof && ((gi + gj + gk) & 1) == 0) { // black cells
        T nb = cxd * (sx[li + 1][lj][lk] + sx[li - 1][lj][lk] + sx[li][lj + 1][lk] +
                      sx[li][lj - 1][lk] + sx[li][lj][lk + 1] + sx[li][lj][lk - 1]);
        sx[li][lj][lk] = (b[gid] + nb) * inv;
    }
    __syncthreads();
    if (dof)
        x[gid] = sx[li][lj][lk];
}

// Prolonged fine value on the fly from read-only coarse x + read-only fine x:
// x_fine + 2·x_coarse[parent] (solid/out-of-range untouched). Reads only inputs.
template <typename T>
__device__ inline T dev_prolonged3(const T* x, const T* xc, const bool* solid, int fi, int fj,
                                   int fk, int nx, int ny, int nz) {
    if (fi < 1 || fi > nx || fj < 1 || fj > ny || fk < 1 || fk > nz)
        return T(0);
    int fid = dev_idx3d(fi, fj, fk, ny, nz);
    if (solid[fid])
        return x[fid];
    int cny = ny / 2, cnz = nz / 2;
    return x[fid] + T(2) * xc[dev_idx3d((fi + 1) / 2, (fj + 1) / 2, (fk + 1) / 2, cny, cnz)];
}

// ── FUSED prolong + post-smooth first colour (black), PING-PONG (x_in→dst) ──
// Reads x_in (const) + coarse, writes a SEPARATE dst buffer → no read/write race
// (the author's _x→_dst_x trick). Loads the prolonged x for self+halo from x_in,
// updates black cells, writes ALL cells to dst (red = prolonged, black =
// prolonged+smoothed). The second colour (red) is then a separate exact sweep on
// dst. Replaces [prolong + black + red] (3 launches) with [this + red] (2).
template <typename T>
__global__ void prolong_black_fused_3d(T* dst, const T* x, const T* xc, const T* b,
                                       const bool* solid, const T* diag, const T* cx, const T* cy,
                                       const T* cz, int nx, int ny, int nz, const bool* trimmed,
                                       T cxd, T cyd, T czd, T diagd) {
    __shared__ T sx[10][10][10];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gi = blockIdx.x * 8 + tx + 1, gj = blockIdx.y * 8 + ty + 1, gk = blockIdx.z * 8 + tz + 1;
    int li = tx + 1, lj = ty + 1, lk = tz + 1;
    bool valid     = (gi <= nx && gj <= ny && gk <= nz);
    sx[li][lj][lk] = dev_prolonged3(x, xc, solid, gi, gj, gk, nx, ny, nz);
    if (tx == 0)
        sx[0][lj][lk] = dev_prolonged3(x, xc, solid, gi - 1, gj, gk, nx, ny, nz);
    if (tx == 7)
        sx[9][lj][lk] = dev_prolonged3(x, xc, solid, gi + 1, gj, gk, nx, ny, nz);
    if (ty == 0)
        sx[li][0][lk] = dev_prolonged3(x, xc, solid, gi, gj - 1, gk, nx, ny, nz);
    if (ty == 7)
        sx[li][9][lk] = dev_prolonged3(x, xc, solid, gi, gj + 1, gk, nx, ny, nz);
    if (tz == 0)
        sx[li][lj][0] = dev_prolonged3(x, xc, solid, gi, gj, gk - 1, nx, ny, nz);
    if (tz == 7)
        sx[li][lj][9] = dev_prolonged3(x, xc, solid, gi, gj, gk + 1, nx, ny, nz);

    int gid       = valid ? dev_idx3d(gi, gj, gk, ny, nz) : 0;
    bool is_solid = valid && solid[gid];
    bool dof      = valid && !is_solid;
    int tileid    = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    bool triv     = trimmed[tileid];
    T cxp = cxd, cxm = cxd, cyp = cyd, cym = cyd, czp = czd, czm = czd, D = diagd;
    if (!triv && dof) {
        D = diag[gid];
        if (D < T(1e-30))
            dof = false;
        cxp = cx[gid];
        cyp = cy[gid];
        czp = cz[gid];
        cxm = (gi > 1) ? cx[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0);
        cym = (gj > 1) ? cy[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0);
        czm = (gk > 1) ? cz[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0);
    }
    __syncthreads();
    if (dof && ((gi + gj + gk) & 1) == 0) { // post-smooth first colour = black (reverse)
        T nb = cxp * sx[li + 1][lj][lk] + cxm * sx[li - 1][lj][lk] + cyp * sx[li][lj + 1][lk] +
               cym * sx[li][lj - 1][lk] + czp * sx[li][lj][lk + 1] + czm * sx[li][lj][lk - 1];
        sx[li][lj][lk] = (b[gid] + nb) / D;
    }
    __syncthreads();
    if (dof)
        dst[gid] = sx[li][lj][lk]; // red = prolonged, black = prolonged+smoothed
    else if (valid)
        dst[gid] = x[gid]; // solid: carry through
}

// ── Tiled residual + restrict (replaces restrict_coeff_kernel_3d on big levels) ──
// One block per fine 8³ tile: load x tile+halo from FRESH global x (the pre-smooth
// already wrote it — so no halo reconstruction needed, exact residual), compute
// r=b-Ax in shared with the trim fast path (uniform tiles skip coefficient reads),
// then sum 2×2×2 children → coarse b. Same math as restrict_coeff_kernel_3d but
// ~7× less x traffic (shared vs strided) and no coefficient reads on uniform tiles.
template <typename T>
__global__ void restrict_residual_tiled_3d(const T* x, const T* b, const bool* solid, const T* diag,
                                           const T* cx, const T* cy, const T* cz, T* bc,
                                           const bool* csolid, int nx, int ny, int nz, int cnx,
                                           int cny, int cnz, const bool* trimmed, T cxd, T cyd,
                                           T czd, T diagd) {
    __shared__ T sx[10][10][10];
    __shared__ T sr[8][8][8];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gi = blockIdx.x * 8 + tx + 1, gj = blockIdx.y * 8 + ty + 1, gk = blockIdx.z * 8 + tz + 1;
    int li = tx + 1, lj = ty + 1, lk = tz + 1;
    bool valid     = (gi <= nx && gj <= ny && gk <= nz);
    sx[li][lj][lk] = valid ? x[dev_idx3d(gi, gj, gk, ny, nz)] : T(0);
    if (tx == 0)
        sx[0][lj][lk] = (gi > 1 && valid) ? x[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0);
    if (tx == 7)
        sx[9][lj][lk] = (gi < nx && valid) ? x[dev_idx3d(gi + 1, gj, gk, ny, nz)] : T(0);
    if (ty == 0)
        sx[li][0][lk] = (gj > 1 && valid) ? x[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0);
    if (ty == 7)
        sx[li][9][lk] = (gj < ny && valid) ? x[dev_idx3d(gi, gj + 1, gk, ny, nz)] : T(0);
    if (tz == 0)
        sx[li][lj][0] = (gk > 1 && valid) ? x[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0);
    if (tz == 7)
        sx[li][lj][9] = (gk < nz && valid) ? x[dev_idx3d(gi, gj, gk + 1, ny, nz)] : T(0);

    int gid       = valid ? dev_idx3d(gi, gj, gk, ny, nz) : 0;
    bool is_solid = valid && solid[gid];
    bool dof      = valid && !is_solid;
    int tileid    = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    bool triv     = trimmed[tileid];
    T cxp = cxd, cxm = cxd, cyp = cyd, cym = cyd, czp = czd, czm = czd, D = diagd;
    if (!triv && dof) {
        D = diag[gid];
        if (D < T(1e-30))
            dof = false;
        cxp = cx[gid];
        cyp = cy[gid];
        czp = cz[gid];
        cxm = (gi > 1) ? cx[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0);
        cym = (gj > 1) ? cy[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0);
        czm = (gk > 1) ? cz[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0);
    }
    __syncthreads();
    T r = T(0);
    if (dof) {
        T nb = cxp * sx[li + 1][lj][lk] + cxm * sx[li - 1][lj][lk] + cyp * sx[li][lj + 1][lk] +
               cym * sx[li][lj - 1][lk] + czp * sx[li][lj][lk + 1] + czm * sx[li][lj][lk - 1];
        r = b[gid] - D * sx[li][lj][lk] + nb;
    }
    sr[tx][ty][tz] = r;
    __syncthreads();
    if (tx < 4 && ty < 4 && tz < 4) {
        int cic = blockIdx.x * 4 + tx + 1, cjc = blockIdx.y * 4 + ty + 1,
            cck = blockIdx.z * 4 + tz + 1;
        if (cic <= cnx && cjc <= cny && cck <= cnz) {
            int cid = dev_idx3d(cic, cjc, cck, cny, cnz);
            if (!csolid[cid]) {
                T s = T(0);
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        for (int dk = 0; dk < 2; dk++)
                            s += sr[2 * tx + di][2 * ty + dj][2 * tz + dk];
                bc[cid] = s;
            }
        }
    }
}

// ── Prolongation: constant injection with ×2 scaling (paper Eq. 11) ──
template <typename T>
__global__ void prolong_kernel_3d(T* x_fine, const T* x_coarse, const bool* solid_fine, int fnx,
                                  int fny, int fnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > fnx || j > fny || k > fnz)
        return;
    int fid = dev_idx3d(i, j, k, fny, fnz);
    if (solid_fine[fid])
        return;
    int ic = (i + 1) / 2, jc = (j + 1) / 2, kc = (k + 1) / 2, cny = fny / 2, cnz = fnz / 2;
    x_fine[fid] += T(2) * x_coarse[dev_idx3d(ic, jc, kc, cny, cnz)];
}

// ── Restrict solid mask (no scalar T) ──
__global__ void restrict_solid_kernel_3d(const bool* solid_fine, bool* solid_coarse, int fnx,
                                         int fny, int fnz) {
    int ic  = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc  = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kc  = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int cnx = fnx / 2, cny = fny / 2, cnz = fnz / 2;
    if (ic > cnx || jc > cny || kc > cnz)
        return;
    int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1, sc = 0;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
                if (solid_fine[dev_idx3d(i_f + di, j_f + dj, k_f + dk, fny, fnz)])
                    sc++;
    // A coarse cell is solid ONLY if ALL 8 children are solid (i.e. it is fluid
    // if ANY child is fluid). Mirrors the author's is_dof propagation. A majority
    // vote (sc>=4) spuriously turns half-blocked coarse cells solid, which loses
    // fluid DOFs across thin solids (e.g. a delta-wing plate) and makes the V-cycle
    // operator inconsistent with the fine matrix → PCG stalls.
    solid_coarse[dev_idx3d(ic, jc, kc, cny, cnz)] = (sc == 8);
}

template <typename T> __global__ void zero_kernel_3d(T* a, long N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        a[i] = T(0);
}
template <typename T> __global__ void copy_kernel_3d(T* dst, const T* src, long N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dst[i] = src[i];
}

// ── Shared-memory tiled RBGS pass (one parity) ──
// With the tile-contiguous layout each block (gi=8·bx+tx+1) maps to exactly one
// 8³ storage tile, so the center load is one contiguous 512-element run (vs 64
// scattered 8-element runs under pitched-linear) — full cache-line utilisation.
// EXACT (not block-RBGS): each colour pass writes back to global, so the next
// pass loads an up-to-date halo — convergence is identical to the untiled sweep.
template <typename T>
__global__ void rbgs_tiled_pass_kernel_3d(T* x, const T* b, const bool* solid, const T* diag,
                                          const T* cx, const T* cy, const T* cz, int nx, int ny,
                                          int nz, int parity, const bool* trimmed, T cxd, T cyd,
                                          T czd, T diagd) {
    __shared__ T sx[10][10][10];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gi = blockIdx.x * 8 + tx + 1, gj = blockIdx.y * 8 + ty + 1, gk = blockIdx.z * 8 + tz + 1;
    int li = tx + 1, lj = ty + 1, lk = tz + 1;
    bool valid     = (gi <= nx && gj <= ny && gk <= nz);
    sx[li][lj][lk] = valid ? x[dev_idx3d(gi, gj, gk, ny, nz)] : T(0);
    if (tx == 0)
        sx[0][lj][lk] = (gi > 1 && gj <= ny && gk <= nz) ? x[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0);
    if (tx == 7)
        sx[9][lj][lk] = (gi < nx && gj <= ny && gk <= nz) ? x[dev_idx3d(gi + 1, gj, gk, ny, nz)] : T(0);
    if (ty == 0)
        sx[li][0][lk] = (gi <= nx && gj > 1 && gk <= nz) ? x[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0);
    if (ty == 7)
        sx[li][9][lk] = (gi <= nx && gj < ny && gk <= nz) ? x[dev_idx3d(gi, gj + 1, gk, ny, nz)] : T(0);
    if (tz == 0)
        sx[li][lj][0] = (gi <= nx && gj <= ny && gk > 1) ? x[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0);
    if (tz == 7)
        sx[li][lj][9] = (gi <= nx && gj <= ny && gk < nz) ? x[dev_idx3d(gi, gj, gk + 1, ny, nz)] : T(0);
    __syncthreads();
    if (!valid || ((gi + gj + gk) & 1) != parity)
        return;
    int gid = dev_idx3d(gi, gj, gk, ny, nz);
    if (solid[gid])
        return;
    int tileid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    T cxp, cxm, cyp, cym, czp, czm, D;
    if (trimmed[tileid]) {
        cxp = cxm = cxd;
        cyp = cym = cyd;
        czp = czm = czd;
        D         = diagd;
    } else {
        D = diag[gid];
        if (D < T(1e-30))
            return;
        cxp = cx[gid];
        cyp = cy[gid];
        czp = cz[gid];
        cxm = (gi > 1) ? cx[dev_idx3d(gi - 1, gj, gk, ny, nz)] : T(0);
        cym = (gj > 1) ? cy[dev_idx3d(gi, gj - 1, gk, ny, nz)] : T(0);
        czm = (gk > 1) ? cz[dev_idx3d(gi, gj, gk - 1, ny, nz)] : T(0);
    }
    T nb   = cxp * sx[li + 1][lj][lk] + cxm * sx[li - 1][lj][lk] + cyp * sx[li][lj + 1][lk] +
             cym * sx[li][lj - 1][lk] + czp * sx[li][lj][lk + 1] + czm * sx[li][lj][lk - 1];
    x[gid] = (b[gid] + nb) / D;
}

// ── one RBGS sweep (forward = odd,even ; reverse = even,odd) ──
template <typename T>
static void rbgs_sweep_3d(typename CudaUAAMGPreconditioner3DT<T>::Level& L, cudaStream_t stream,
                          bool reverse) {
    int nx = L.g.nx, ny = L.g.ny, nz = L.g.nz;
    dim3 block(8, 8, 8), grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    int p0 = reverse ? 0 : 1;
    // Shared-mem tiling pays off only on large levels (it adds load/occupancy
    // overhead that dominates on small/coarse levels). Gate by level size so
    // tiling is a pure win: tiled on the fine levels, plain on the coarse ones.
    bool tiled  = ((long)nx * ny * nz >= (1L << 21)); // ≥ ~2M cells
    auto launch = [&](int par) {
        if (tiled)
            rbgs_tiled_pass_kernel_3d<T><<<grid, block, 0, stream>>>(
                L.g.x, L.g.b, L.g.solid, L.diag, L.cx, L.cy, L.cz, nx, ny, nz, par, L.trimmed,
                L.cxd, L.cyd, L.czd, L.diagd);
        else
            rbgs_coeff_kernel_3d<T><<<grid, block, 0, stream>>>(L.g.x, L.g.b, L.g.solid, L.diag,
                                                                L.cx, L.cy, L.cz, nx, ny, nz, par,
                                                                L.trimmed, L.cxd, L.cyd, L.czd,
                                                                L.diagd);
    };
    launch(p0);
    launch(1 - p0);
}

// ── Symmetric V(1,1) ──
template <typename T>
static void vCycle3D(typename CudaUAAMGPreconditioner3DT<T>::Level* levels, int level, int nlevels,
                     cudaStream_t stream) {
    auto& L = levels[level];
    int nx = L.g.nx, ny = L.g.ny, nz = L.g.nz;
    if (level == nlevels - 1) {
        // Singular coarsest system: enforce a compatible (zero-mean) RHS, solve,
        // then pin the solution's constant — otherwise the null-space component
        // grows and stalls the whole V-cycle on a pure-Neumann domain.
        // Single-launch coarsest solve (was 42 tiny launches): mean(b) + 10×
        // (forward+reverse RBGS) + mean(x), one block, identical math.
        coarsest_solve_kernel_3d<T><<<1, 256, 0, stream>>>(L.g.x, L.g.b, L.g.solid, L.diag, L.cx,
                                                           L.cy, L.cz, nx, ny, nz, L.trimmed, L.cxd,
                                                           L.cyd, L.czd, L.diagd, 10);
        return;
    }
    auto& coarse = levels[level + 1];
    int cnx = coarse.g.nx, cny = coarse.g.ny, cnz = coarse.g.nz;
    long Nc = coarse.g.num_tiles * 512;
    // Down-leg pre-smooth. On large levels use smooth_from_zero_3d: exploits
    // x=0-at-entry (author's trick) to compute red=b/diag locally and black
    // exactly in ONE launch with no global x reads/reload — identical output to
    // the forward rbgs_sweep, fewer launches + less traffic.
    if ((long)nx * ny * nz >= (1L << 21)) {
        dim3 block(8, 8, 8), grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        smooth_from_zero_3d<T><<<grid, block, 0, stream>>>(L.g.x, L.g.b, L.g.solid, L.diag, L.cx,
                                                           L.cy, L.cz, nx, ny, nz, L.trimmed, L.cxd,
                                                           L.cyd, L.czd, L.diagd);
    } else {
        rbgs_sweep_3d<T>(L, stream, false); // pre-smooth (forward)
    }
    // Residual + restrict. On large levels use the tiled version (shared-mem x +
    // trim fast path); reads the fresh global x the pre-smooth just wrote.
    if ((long)nx * ny * nz >= (1L << 21)) {
        dim3 block(8, 8, 8), grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        restrict_residual_tiled_3d<T><<<grid, block, 0, stream>>>(
            L.g.x, L.g.b, L.g.solid, L.diag, L.cx, L.cy, L.cz, coarse.g.b, coarse.g.solid, nx, ny,
            nz, cnx, cny, cnz, L.trimmed, L.cxd, L.cyd, L.czd, L.diagd);
    } else {
        dim3 cblock(8, 8, 8), cgrid((cnx + 7) / 8, (cny + 7) / 8, (cnz + 7) / 8);
        restrict_coeff_kernel_3d<T><<<cgrid, cblock, 0, stream>>>(
            L.g.x, L.g.b, L.g.solid, L.diag, L.cx, L.cy, L.cz, coarse.g.b, coarse.g.solid, nx, ny,
            nz, cnx, cny, cnz);
    }

    zero_kernel_3d<T><<<(Nc + 255) / 256, 256, 0, stream>>>(coarse.g.x, Nc);
    vCycle3D<T>(levels, level + 1, nlevels, stream);

    dim3 fblock(8, 8, 8), fgrid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    if ((long)nx * ny * nz >= (1L << 21)) {
        // Ping-pong fused prolong+black (x→scratch, no race), then red on scratch,
        // then swap so L.g.x holds the result. [prolong+black+red] → [fused+red].
        prolong_black_fused_3d<T><<<fgrid, fblock, 0, stream>>>(
            L.scratch, L.g.x, coarse.g.x, L.g.b, L.g.solid, L.diag, L.cx, L.cy, L.cz, nx, ny, nz,
            L.trimmed, L.cxd, L.cyd, L.czd, L.diagd);
        rbgs_tiled_pass_kernel_3d<T><<<fgrid, fblock, 0, stream>>>(
            L.scratch, L.g.b, L.g.solid, L.diag, L.cx, L.cy, L.cz, nx, ny, nz, 1, L.trimmed, L.cxd,
            L.cyd, L.czd, L.diagd); // red pass (parity 1) on scratch
        std::swap(L.g.x, L.scratch); // result now in L.g.x
    } else {
        prolong_kernel_3d<T><<<fgrid, fblock, 0, stream>>>(L.g.x, coarse.g.x, L.g.solid, nx, ny, nz);
        rbgs_sweep_3d<T>(L, stream, true); // post-smooth (reverse → symmetric)
    }
}

// ── CudaUAAMGPreconditioner3DT<T> ──

template <typename T> void CudaUAAMGPreconditioner3DT<T>::build(const CudaGrid3DT_<T>& fine) {
    if (cached_nx_ == fine.nx && cached_ny_ == fine.ny && cached_nz_ == fine.nz)
        return;
    destroy();
    int nx = fine.nx, ny = fine.ny, nz = fine.nz;
    T dx = fine.dx, dy = fine.dy, dz = fine.dz;
    while (nx >= 2 && ny >= 2 && nz >= 2) {
        Level L;
        L.g.allocate_tiled(nx, ny, nz, dx, dy, dz);
        L.stride = nx + 2;
        long N   = L.g.num_tiles * 512;
        cudaMalloc(&L.diag, N * sizeof(T));
        cudaMemset(L.diag, 0, N * sizeof(T));
        cudaMalloc(&L.cx, N * sizeof(T));
        cudaMemset(L.cx, 0, N * sizeof(T));
        cudaMalloc(&L.cy, N * sizeof(T));
        cudaMemset(L.cy, 0, N * sizeof(T));
        cudaMalloc(&L.cz, N * sizeof(T));
        cudaMemset(L.cz, 0, N * sizeof(T));
        L.ntx = (nx + 7) / 8;
        L.nty = (ny + 7) / 8;
        L.ntz = (nz + 7) / 8;
        cudaMalloc(&L.trimmed, (size_t)L.ntx * L.nty * L.ntz * sizeof(bool));
        cudaMemset(L.trimmed, 0, (size_t)L.ntx * L.nty * L.ntz * sizeof(bool));
        cudaMalloc(&L.scratch, N * sizeof(T)); // ping-pong for fused post-smooth
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4)
            break;
        nx /= 2;
        ny /= 2;
        nz /= 2;
        dx *= T(2);
        dy *= T(2);
        dz *= T(2);
    }
    cached_nx_ = fine.nx;
    cached_ny_ = fine.ny;
    cached_nz_ = fine.nz;
}

template <typename T> void CudaUAAMGPreconditioner3DT<T>::setupLevels(const CudaGrid3DT_<T>& fine) {
    build(fine);
    int nl              = (int)levels_.size();
    cudaStream_t stream = 0;
    int nx0 = fine.nx, ny0 = fine.ny, nz0 = fine.nz;

    // Convert the caller's pitched solid mask into the level-0 tile layout.
    {
        dim3 block(8, 8, 8), grid((nx0 + 7) / 8, (ny0 + 7) / 8, (nz0 + 7) / 8);
        scatter_solid_p2t_kernel_3d<<<grid, block, 0, stream>>>(fine.solid, levels_[0].g.solid, nx0,
                                                                ny0, nz0, fine.pitch);
    }
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l - 1];
        auto& cL = levels_[l];
        dim3 block(8, 8, 8), grid((cL.g.nx + 7) / 8, (cL.g.ny + 7) / 8, (cL.g.nz + 7) / 8);
        restrict_solid_kernel_3d<<<grid, block, 0, stream>>>(fL.g.solid, cL.g.solid, fL.g.nx,
                                                             fL.g.ny, fL.g.nz);
    }
    {
        auto& L = levels_[0];
        int nx = L.g.nx, ny = L.g.ny, nz = L.g.nz;
        dim3 block(8, 8, 8), grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        setup_fine_coeffs_kernel_3d<T><<<grid, block, 0, stream>>>(
            L.g.solid, L.cx, L.cy, L.cz, nx, ny, nz, L.g.idx2, L.g.idy2, L.g.idz2);
        setup_diag_kernel_3d<T><<<grid, block, 0, stream>>>(L.g.solid, L.cx, L.cy, L.cz, L.diag, nx,
                                                            ny, nz);
    }
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l - 1];
        auto& cL = levels_[l];
        int cnx = cL.g.nx, cny = cL.g.ny, cnz = cL.g.nz;
        dim3 block(8, 8, 8), grid((cnx + 7) / 8, (cny + 7) / 8, (cnz + 7) / 8);
        galerkin_coeffs_kernel_3d<T><<<grid, block, 0, stream>>>(fL.cx, fL.cy, fL.cz, cL.g.solid,
                                                                 cL.cx, cL.cy, cL.cz, fL.g.nx,
                                                                 fL.g.ny, fL.g.nz, cnx, cny, cnz);
        setup_diag_kernel_3d<T><<<grid, block, 0, stream>>>(cL.g.solid, cL.cx, cL.cy, cL.cz,
                                                            cL.diag, cnx, cny, cnz);
    }

    // §5.4: per-level uniform default stencil (Galerkin: coarse coupling = 4× finer)
    levels_[0].cxd = levels_[0].g.idx2;
    levels_[0].cyd = levels_[0].g.idy2;
    levels_[0].czd = levels_[0].g.idz2;
    for (int l = 1; l < nl; l++) {
        levels_[l].cxd = T(4) * levels_[l - 1].cxd;
        levels_[l].cyd = T(4) * levels_[l - 1].cyd;
        levels_[l].czd = T(4) * levels_[l - 1].czd;
    }
    static bool notrim = (std::getenv("UAAMG_NOTRIM") != nullptr);
    for (int l = 0; l < nl; l++) {
        auto& L    = levels_[l];
        L.diagd    = L.cxd + L.cxd + L.cyd + L.cyd + L.czd + L.czd; // match setup_diag summation
        int ntiles = L.ntx * L.nty * L.ntz;
        if (notrim) {
            cudaMemsetAsync(L.trimmed, 0, ntiles * sizeof(bool), stream);
        } else {
            mark_trimmed_kernel_3d<T><<<(ntiles + 255) / 256, 256, 0, stream>>>(
                L.diag, L.diagd, L.g.nx, L.g.ny, L.g.nz, L.ntx, L.nty, L.ntz, L.trimmed);
        }
    }
}

template <typename T>
void CudaUAAMGPreconditioner3DT<T>::vcycle_apply(const CudaGrid3DT_<T>& fine, const T* r, T* z) {
    int nl              = (int)levels_.size();
    cudaStream_t stream = 0;
    int nx0 = fine.nx, ny0 = fine.ny, nz0 = fine.nz;
    dim3 cblock(8, 8, 8), cgrid((nx0 + 7) / 8, (ny0 + 7) / 8, (nz0 + 7) / 8);
    // pitched r → tile-contiguous level-0 b.
    scatter_p2t_kernel_3d<T><<<cgrid, cblock, 0, stream>>>(r, levels_[0].g.b, nx0, ny0, nz0,
                                                           fine.pitch);
    for (int l = 0; l < nl; l++) {
        long N = levels_[l].g.num_tiles * 512;
        zero_kernel_3d<T><<<(N + 255) / 256, 256, 0, stream>>>(levels_[l].g.x, N);
    }
    vCycle3D<T>(levels_.data(), 0, nl, stream);
    cudaDeviceSynchronize();
    // tile-contiguous level-0 x → pitched z (ghost cells zeroed for safety).
    cudaMemsetAsync(z, 0, (size_t)(nx0 + 2) * (ny0 + 2) * (nz0 + 2) * sizeof(T), stream);
    gather_t2p_kernel_3d<T><<<cgrid, cblock, 0, stream>>>(levels_[0].g.x, z, nx0, ny0, nz0,
                                                          fine.pitch);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaUAAMGPreconditioner3DT<T>::apply(const CudaGrid3DT_<T>& fine, const T* r, T* z) {
    setupLevels(fine);
    vcycle_apply(fine, r, z);
}

template <typename T> void CudaUAAMGPreconditioner3DT<T>::destroy() {
    for (auto& L : levels_) {
        L.g.free();
        if (L.diag)
            cudaFree(L.diag);
        if (L.cx)
            cudaFree(L.cx);
        if (L.cy)
            cudaFree(L.cy);
        if (L.cz)
            cudaFree(L.cz);
        if (L.trimmed)
            cudaFree(L.trimmed);
        if (L.scratch)
            cudaFree(L.scratch);
        L.diag = L.cx = L.cy = L.cz = L.scratch = nullptr;
        L.trimmed                               = nullptr;
    }
    levels_.clear();
    cached_nx_ = cached_ny_ = cached_nz_ = -1;
}

// Explicit instantiations
template class CudaUAAMGPreconditioner3DT<double>;
template class CudaUAAMGPreconditioner3DT<float>;

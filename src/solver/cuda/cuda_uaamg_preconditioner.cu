/**
 * @file cuda_uaamg_preconditioner.cu
 * @brief CUDA UAAMG V-cycle — Algorithm 3, Section 5.1.
 *
 * Sun et al., "Leapfrog Flow Maps for Real-Time Fluid Simulation", SIGGRAPH 2025.
 *
 * Design decisions for CPU/GPU numerical consistency:
 *  - RBGS split into red-pass + black-pass kernels (global sync between passes).
 *    This avoids cross-tile ordering issues and matches CPU RBGS iteration order.
 *  - No shared-memory tiling → simpler, bit-identical to CPU per cell.
 *  - Per-fine-cell prolongation (no atomics, paper Section 5.3).
 *  - Scale-2 prolongation (Stuben 2001).
 *  - Matrix-free: stencil coefficients from dx, dy per level.
 */
#include "solver/cuda/cuda_uaamg_preconditioner.h"

// ── Column-major indexing (matches Grid::ip) ──
__device__ inline int dev_idx(int i, int j, int stride) { return i + j * stride; }

// ─────────────────────────────────────────────────────────
//  Kernel: RBGS first pass  (i+j odd, matches CPU checkerboard order)
// ─────────────────────────────────────────────────────────
__global__ void rbgs_pass1_kernel(
    double *x, const double *b, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    if (!((i + j) & 1)) return;  // only odd sum (matches CPU first pass)
    int id = dev_idx(i, j, stride);
    if (solid[id]) return;

    double pC = x[id];
    double pL = (i>1 && !solid[dev_idx(i-1,j,stride)]) ? x[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !solid[dev_idx(i+1,j,stride)]) ? x[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !solid[dev_idx(i,j-1,stride)]) ? x[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !solid[dev_idx(i,j+1,stride)]) ? x[dev_idx(i,j+1,stride)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
    double eff_d = diag;
    if (i==1||solid[dev_idx(i-1,j,stride)]) eff_d -= idx2;
    if (i==nx||solid[dev_idx(i+1,j,stride)]) eff_d -= idx2;
    if (j==1||solid[dev_idx(i,j-1,stride)]) eff_d -= idy2;
    if (j==ny||solid[dev_idx(i,j+1,stride)]) eff_d -= idy2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

// ─────────────────────────────────────────────────────────
//  Kernel: RBGS second pass (i+j even, neighbors = odd, just updated)
// ─────────────────────────────────────────────────────────
__global__ void rbgs_pass2_kernel(
    double *x, const double *b, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    if ((i + j) & 1) return;  // only even sum (matches CPU second pass)
    int id = dev_idx(i, j, stride);
    if (solid[id]) return;

    double pC = x[id];
    double pL = (i>1 && !solid[dev_idx(i-1,j,stride)]) ? x[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !solid[dev_idx(i+1,j,stride)]) ? x[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !solid[dev_idx(i,j-1,stride)]) ? x[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !solid[dev_idx(i,j+1,stride)]) ? x[dev_idx(i,j+1,stride)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
    double eff_d = diag;
    if (i==1||solid[dev_idx(i-1,j,stride)]) eff_d -= idx2;
    if (i==nx||solid[dev_idx(i+1,j,stride)]) eff_d -= idx2;
    if (j==1||solid[dev_idx(i,j-1,stride)]) eff_d -= idy2;
    if (j==ny||solid[dev_idx(i,j+1,stride)]) eff_d -= idy2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

// ─────────────────────────────────────────────────────────
//  Kernel: Restrict residual (4-to-1 averaging)
// ─────────────────────────────────────────────────────────
__global__ void restrict_kernel(
    const double *x_fine, const double *b_fine, const bool *solid_fine,
    double *b_coarse, bool *solid_coarse,
    int fnx, int fny, int fstride, int cstride,
    double idx2, double idy2, double diag)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int cnx = fnx / 2, cny = fny / 2;
    if (ic > cnx || jc > cny) return;

    int i_f = 2*ic - 1, j_f = 2*jc - 1;
    double sum = 0; int cnt = 0;
    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            int fi = i_f + di, fj = j_f + dj;
            int fidx = dev_idx(fi, fj, fstride);
            if (solid_fine[fidx]) continue;
            double pC = x_fine[fidx];
            double pL = (fi>1 && !solid_fine[dev_idx(fi-1,fj,fstride)]) ? x_fine[dev_idx(fi-1,fj,fstride)] : pC;
            double pR = (fi<fnx && !solid_fine[dev_idx(fi+1,fj,fstride)]) ? x_fine[dev_idx(fi+1,fj,fstride)] : pC;
            double pB = (fj>1 && !solid_fine[dev_idx(fi,fj-1,fstride)]) ? x_fine[dev_idx(fi,fj-1,fstride)] : pC;
            double pT = (fj<fny && !solid_fine[dev_idx(fi,fj+1,fstride)]) ? x_fine[dev_idx(fi,fj+1,fstride)] : pC;
            double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
            sum += b_fine[fidx] - diag * pC + lap;
            cnt++;
        }
    }
    int cid = dev_idx(ic, jc, cstride);
    if (!solid_coarse[cid] && cnt > 0)
        b_coarse[cid] = sum / cnt;
}

// ─────────────────────────────────────────────────────────
//  Kernel: Prolongation — per-fine-cell, no atomics (paper)
// ─────────────────────────────────────────────────────────
__global__ void prolong_kernel(
    double *x_fine, const double *x_coarse, const bool *solid_fine,
    int fnx, int fny, int fstride, int cstride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > fnx || j > fny) return;
    int fid = dev_idx(i, j, fstride);
    if (solid_fine[fid]) return;

    int ic = (i + 1) / 2;
    int jc = (j + 1) / 2;
    x_fine[fid] += 2.0 * x_coarse[dev_idx(ic, jc, cstride)];  // scale 2
}

// ─────────────────────────────────────────────────────────
//  Kernel: Restrict solid mask
// ─────────────────────────────────────────────────────────
__global__ void restrict_solid_kernel(
    const bool *solid_fine, bool *solid_coarse,
    int fnx, int fny, int fstride, int cstride)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int cnx = fnx / 2, cny = fny / 2;
    if (ic > cnx || jc > cny) return;

    int i_f = 2*ic - 1, j_f = 2*jc - 1, sc = 0;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            if (solid_fine[dev_idx(i_f+di, j_f+dj, fstride)]) sc++;
    solid_coarse[dev_idx(ic, jc, cstride)] = (sc >= 2);
}

// ─────────────────────────────────────────────────────────
//  Kernel: Memset
// ─────────────────────────────────────────────────────────
__global__ void zero_kernel(double *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = 0.0;
}

__global__ void copy_kernel(double *dst, const double *src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ─────────────────────────────────────────────────────────
//  Host: one RBGS sweep (red + black, two kernel launches)
// ─────────────────────────────────────────────────────────
static void rbgs_sweep(CudaUAAMGPreconditioner::Level& L, cudaStream_t stream) {
    int nx = L.g.nx, ny = L.g.ny;
    dim3 block(16, 16);
    dim3 grid((nx + 15)/16, (ny + 15)/16);

    // Order matches CPU: pass1 = odd-sum first, pass2 = even-sum second
    rbgs_pass1_kernel<<<grid, block, 0, stream>>>(
        L.g.x, L.g.b, L.g.solid, nx, ny, L.g.pitch,
        L.g.idx2, L.g.idy2, L.g.diag);
    rbgs_pass2_kernel<<<grid, block, 0, stream>>>(
        L.g.x, L.g.b, L.g.solid, nx, ny, L.g.pitch,
        L.g.idx2, L.g.idy2, L.g.diag);
}

// ─────────────────────────────────────────────────────────
//  Recursive V-Cycle (Algorithm 3)
// ─────────────────────────────────────────────────────────
static void vCycle(CudaUAAMGPreconditioner::Level* levels, int level, int nlevels,
                   cudaStream_t stream)
{
    auto& L = levels[level];
    int nx = L.g.nx, ny = L.g.ny;

    if (level == nlevels - 1) {
        for (int s = 0; s < 20; s++) rbgs_sweep(L, stream);
        return;
    }

    // Pre-smooth
    rbgs_sweep(L, stream);

    // Restrict
    auto& coarse = levels[level + 1];
    int cnx = coarse.g.nx, cny = coarse.g.ny;
    dim3 cblock(16, 16);
    dim3 cgrid((cnx + 15)/16, (cny + 15)/16);
    restrict_kernel<<<cgrid, cblock, 0, stream>>>(
        L.g.x, L.g.b, L.g.solid, coarse.g.b, coarse.g.solid,
        nx, ny, L.g.pitch, coarse.g.pitch,
        L.g.idx2, L.g.idy2, L.g.diag);

    // Zero coarse x & recurse
    int Nc = (cnx+2)*(cny+2);
    zero_kernel<<<(Nc+255)/256, 256, 0, stream>>>(coarse.g.x, Nc);
    vCycle(levels, level + 1, nlevels, stream);

    // Prolongate
    dim3 fblock(16, 16);
    dim3 fgrid((nx + 15)/16, (ny + 15)/16);
    prolong_kernel<<<fgrid, fblock, 0, stream>>>(
        L.g.x, coarse.g.x, L.g.solid, nx, ny, L.g.pitch, coarse.g.pitch);

    // Post-smooth
    rbgs_sweep(L, stream);
}

// ── CudaUAAMGPreconditioner ──

void CudaUAAMGPreconditioner::build(const CudaGrid& fine) {
    if (cached_nx_ == fine.nx && cached_ny_ == fine.ny) return;
    destroy();
    int nx = fine.nx, ny = fine.ny;
    double dx = fine.dx, dy = fine.dy;
    while (nx >= 2 && ny >= 2) {
        Level L;
        L.g.allocate(nx, ny, dx, dy);
        L.stride = nx + 2;
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4) break;
        nx /= 2; ny /= 2; dx *= 2.0; dy *= 2.0;
    }
    cached_nx_ = fine.nx; cached_ny_ = fine.ny;
}

void CudaUAAMGPreconditioner::apply(const CudaGrid& fine, const double* r, double* z) {
    build(fine);
    int nl = (int)levels_.size();
    cudaStream_t stream = 0;
    int N0 = (fine.nx+2)*(fine.ny+2);

    // Copy solid to finest
    cudaMemcpy(levels_[0].g.solid, fine.solid, N0 * sizeof(bool), cudaMemcpyDeviceToDevice);

    // Propagate solid down
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l-1], &cL = levels_[l];
        dim3 block(16, 16);
        dim3 grid((cL.g.nx + 15)/16, (cL.g.ny + 15)/16);
        restrict_solid_kernel<<<grid, block, 0, stream>>>(
            fL.g.solid, cL.g.solid, fL.g.nx, fL.g.ny, fL.g.pitch, cL.g.pitch);
    }

    // Copy r → finest b
    copy_kernel<<<(N0+255)/256, 256, 0, stream>>>(levels_[0].g.b, r, N0);

    // Zero all pressures
    for (int l = 0; l < nl; l++) {
        int N = (levels_[l].g.nx+2)*(levels_[l].g.ny+2);
        zero_kernel<<<(N+255)/256, 256, 0, stream>>>(levels_[l].g.x, N);
    }

    vCycle(levels_.data(), 0, nl, stream);
    cudaDeviceSynchronize();

    // Copy finest correction → z
    copy_kernel<<<(N0+255)/256, 256, 0, stream>>>(z, levels_[0].g.x, N0);
    cudaDeviceSynchronize();
}

void CudaUAAMGPreconditioner::destroy() {
    for (auto& L : levels_) L.g.free();
    levels_.clear();
    cached_nx_ = cached_ny_ = -1;
}

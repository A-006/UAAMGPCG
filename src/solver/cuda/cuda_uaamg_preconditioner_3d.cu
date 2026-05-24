/**
 * @file cuda_uaamg_preconditioner_3d.cu
 * @brief CUDA 3D UAAMG V-cycle preconditioner.
 *
 * 3D extension with 7-point stencil, 8-to-1 restriction (2x2x2),
 * and scale-2 trilinear prolongation.
 *
 * Design (matching 2D):
 *  - RBGS split into odd-pass + even-pass kernels ((i+j+k) parity).
 *  - Per-fine-cell prolongation (no atomics).
 *  - Matrix-free: stencil coefficients from dx, dy, dz per level.
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"

// ── 3D column-major index ──
__device__ inline int dev_idx3d(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

// ─────────────────────────────────────────────────────────────
//  Kernel: RBGS first pass  ((i+j+k) odd)
// ─────────────────────────────────────────────────────────────
__global__ void rbgs_pass1_kernel_3d(
    double *x, const double *b, const bool *solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    if (!((i + j + k) & 1)) return;  // only odd sum
    int id = dev_idx3d(i, j, k, pitch, ny);
    if (solid[id]) return;

    double pC = x[id];
    double pL = (i>1   && !solid[dev_idx3d(i-1, j,   k,   pitch, ny)]) ? x[dev_idx3d(i-1, j,   k,   pitch, ny)] : pC;
    double pR = (i<nx  && !solid[dev_idx3d(i+1, j,   k,   pitch, ny)]) ? x[dev_idx3d(i+1, j,   k,   pitch, ny)] : pC;
    double pB = (j>1   && !solid[dev_idx3d(i,   j-1, k,   pitch, ny)]) ? x[dev_idx3d(i,   j-1, k,   pitch, ny)] : pC;
    double pT = (j<ny  && !solid[dev_idx3d(i,   j+1, k,   pitch, ny)]) ? x[dev_idx3d(i,   j+1, k,   pitch, ny)] : pC;
    double pF = (k>1   && !solid[dev_idx3d(i,   j,   k-1, pitch, ny)]) ? x[dev_idx3d(i,   j,   k-1, pitch, ny)] : pC;
    double pK = (k<nz  && !solid[dev_idx3d(i,   j,   k+1, pitch, ny)]) ? x[dev_idx3d(i,   j,   k+1, pitch, ny)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2 + (pF+pK)*idz2;
    double eff_d = diag;
    if (i==1   || solid[dev_idx3d(i-1, j,   k,   pitch, ny)]) eff_d -= idx2;
    if (i==nx  || solid[dev_idx3d(i+1, j,   k,   pitch, ny)]) eff_d -= idx2;
    if (j==1   || solid[dev_idx3d(i,   j-1, k,   pitch, ny)]) eff_d -= idy2;
    if (j==ny  || solid[dev_idx3d(i,   j+1, k,   pitch, ny)]) eff_d -= idy2;
    if (k==1   || solid[dev_idx3d(i,   j,   k-1, pitch, ny)]) eff_d -= idz2;
    if (k==nz  || solid[dev_idx3d(i,   j,   k+1, pitch, ny)]) eff_d -= idz2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

// ─────────────────────────────────────────────────────────────
//  Kernel: RBGS second pass ((i+j+k) even)
// ─────────────────────────────────────────────────────────────
__global__ void rbgs_pass2_kernel_3d(
    double *x, const double *b, const bool *solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    if ((i + j + k) & 1) return;  // only even sum
    int id = dev_idx3d(i, j, k, pitch, ny);
    if (solid[id]) return;

    double pC = x[id];
    double pL = (i>1   && !solid[dev_idx3d(i-1, j,   k,   pitch, ny)]) ? x[dev_idx3d(i-1, j,   k,   pitch, ny)] : pC;
    double pR = (i<nx  && !solid[dev_idx3d(i+1, j,   k,   pitch, ny)]) ? x[dev_idx3d(i+1, j,   k,   pitch, ny)] : pC;
    double pB = (j>1   && !solid[dev_idx3d(i,   j-1, k,   pitch, ny)]) ? x[dev_idx3d(i,   j-1, k,   pitch, ny)] : pC;
    double pT = (j<ny  && !solid[dev_idx3d(i,   j+1, k,   pitch, ny)]) ? x[dev_idx3d(i,   j+1, k,   pitch, ny)] : pC;
    double pF = (k>1   && !solid[dev_idx3d(i,   j,   k-1, pitch, ny)]) ? x[dev_idx3d(i,   j,   k-1, pitch, ny)] : pC;
    double pK = (k<nz  && !solid[dev_idx3d(i,   j,   k+1, pitch, ny)]) ? x[dev_idx3d(i,   j,   k+1, pitch, ny)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2 + (pF+pK)*idz2;
    double eff_d = diag;
    if (i==1   || solid[dev_idx3d(i-1, j,   k,   pitch, ny)]) eff_d -= idx2;
    if (i==nx  || solid[dev_idx3d(i+1, j,   k,   pitch, ny)]) eff_d -= idx2;
    if (j==1   || solid[dev_idx3d(i,   j-1, k,   pitch, ny)]) eff_d -= idy2;
    if (j==ny  || solid[dev_idx3d(i,   j+1, k,   pitch, ny)]) eff_d -= idy2;
    if (k==1   || solid[dev_idx3d(i,   j,   k-1, pitch, ny)]) eff_d -= idz2;
    if (k==nz  || solid[dev_idx3d(i,   j,   k+1, pitch, ny)]) eff_d -= idz2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

// ─────────────────────────────────────────────────────────────
//  Kernel: Restrict residual (8-to-1, 2x2x2 averaging)
// ─────────────────────────────────────────────────────────────
__global__ void restrict_kernel_3d(
    const double *x_fine, const double *b_fine, const bool *solid_fine,
    double *b_coarse, bool *solid_coarse,
    int fnx, int fny, int fnz, int fpitch, int cpitch,
    double idx2, double idy2, double idz2, double diag)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int cnx = fnx / 2, cny = fny / 2, cnz = fnz / 2;
    if (ic > cnx || jc > cny || kc > cnz) return;

    int i_f = 2*ic - 1, j_f = 2*jc - 1, k_f = 2*kc - 1;
    double sum = 0; int cnt = 0;
    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            for (int dk = 0; dk < 2; dk++) {
                int fi = i_f + di, fj = j_f + dj, fk = k_f + dk;
                int fidx = dev_idx3d(fi, fj, fk, fpitch, fny);
                if (solid_fine[fidx]) continue;

                double pC = x_fine[fidx];
                double pL = (fi>1   && !solid_fine[dev_idx3d(fi-1,fj,  fk,  fpitch,fny)]) ? x_fine[dev_idx3d(fi-1,fj,  fk,  fpitch,fny)] : pC;
                double pR = (fi<fnx && !solid_fine[dev_idx3d(fi+1,fj,  fk,  fpitch,fny)]) ? x_fine[dev_idx3d(fi+1,fj,  fk,  fpitch,fny)] : pC;
                double pB = (fj>1   && !solid_fine[dev_idx3d(fi,  fj-1,fk,  fpitch,fny)]) ? x_fine[dev_idx3d(fi,  fj-1,fk,  fpitch,fny)] : pC;
                double pT = (fj<fny && !solid_fine[dev_idx3d(fi,  fj+1,fk,  fpitch,fny)]) ? x_fine[dev_idx3d(fi,  fj+1,fk,  fpitch,fny)] : pC;
                double pF = (fk>1   && !solid_fine[dev_idx3d(fi,  fj,  fk-1,fpitch,fny)]) ? x_fine[dev_idx3d(fi,  fj,  fk-1,fpitch,fny)] : pC;
                double pK = (fk<fnz && !solid_fine[dev_idx3d(fi,  fj,  fk+1,fpitch,fny)]) ? x_fine[dev_idx3d(fi,  fj,  fk+1,fpitch,fny)] : pC;
                double lap = (pL+pR)*idx2 + (pB+pT)*idy2 + (pF+pK)*idz2;
                sum += b_fine[fidx] - diag * pC + lap;
                cnt++;
            }
        }
    }
    int cid = dev_idx3d(ic, jc, kc, cpitch, cny);
    if (!solid_coarse[cid] && cnt > 0)
        b_coarse[cid] = sum / cnt;
}

// ─────────────────────────────────────────────────────────────
//  Kernel: Prolongation — per-fine-cell, no atomics, scale 2
// ─────────────────────────────────────────────────────────────
__global__ void prolong_kernel_3d(
    double *x_fine, const double *x_coarse, const bool *solid_fine,
    int fnx, int fny, int fnz, int fpitch, int cpitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > fnx || j > fny || k > fnz) return;
    int fid = dev_idx3d(i, j, k, fpitch, fny);
    if (solid_fine[fid]) return;

    int ic = (i + 1) / 2;
    int jc = (j + 1) / 2;
    int kc = (k + 1) / 2;
    int cny = fny / 2;
    x_fine[fid] += 2.0 * x_coarse[dev_idx3d(ic, jc, kc, cpitch, cny)];
}

// ─────────────────────────────────────────────────────────────
//  Kernel: Restrict solid mask (2x2x2 blocks, coarse solid if >=4)
// ─────────────────────────────────────────────────────────────
__global__ void restrict_solid_kernel_3d(
    const bool *solid_fine, bool *solid_coarse,
    int fnx, int fny, int fnz, int fpitch, int cpitch)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int cnx = fnx / 2, cny = fny / 2, cnz = fnz / 2;
    if (ic > cnx || jc > cny || kc > cnz) return;

    int i_f = 2*ic - 1, j_f = 2*jc - 1, k_f = 2*kc - 1, sc = 0;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
                if (solid_fine[dev_idx3d(i_f+di, j_f+dj, k_f+dk, fpitch, fny)]) sc++;
    solid_coarse[dev_idx3d(ic, jc, kc, cpitch, cny)] = (sc >= 4);
}

// ─────────────────────────────────────────────────────────────
//  Utility kernels
// ─────────────────────────────────────────────────────────────
__global__ void zero_kernel_3d(double *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = 0.0;
}

__global__ void copy_kernel_3d(double *dst, const double *src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ─────────────────────────────────────────────────────────────
//  Host: one RBGS sweep (odd + even, two kernel launches)
// ─────────────────────────────────────────────────────────────
static void rbgs_sweep_3d(CudaUAAMGPreconditioner3D::Level& L, cudaStream_t stream) {
    int nx = L.g.nx, ny = L.g.ny, nz = L.g.nz;
    dim3 block(8, 8, 8);
    dim3 grid((nx + 7)/8, (ny + 7)/8, (nz + 7)/8);

    rbgs_pass1_kernel_3d<<<grid, block, 0, stream>>>(
        L.g.x, L.g.b, L.g.solid, nx, ny, nz, L.g.pitch,
        L.g.idx2, L.g.idy2, L.g.idz2, L.g.diag);
    rbgs_pass2_kernel_3d<<<grid, block, 0, stream>>>(
        L.g.x, L.g.b, L.g.solid, nx, ny, nz, L.g.pitch,
        L.g.idx2, L.g.idy2, L.g.idz2, L.g.diag);
}

// ─────────────────────────────────────────────────────────────
//  Recursive V-Cycle (Algorithm 3, 3D version)
// ─────────────────────────────────────────────────────────────
static void vCycle3D(CudaUAAMGPreconditioner3D::Level* levels, int level, int nlevels,
                     cudaStream_t stream)
{
    auto& L = levels[level];
    int nx = L.g.nx, ny = L.g.ny, nz = L.g.nz;

    if (level == nlevels - 1) {
        for (int s = 0; s < 20; s++) rbgs_sweep_3d(L, stream);
        return;
    }

    // Pre-smooth
    rbgs_sweep_3d(L, stream);

    // Restrict
    auto& coarse = levels[level + 1];
    int cnx = coarse.g.nx, cny = coarse.g.ny, cnz = coarse.g.nz;
    dim3 cblock(8, 8, 8);
    dim3 cgrid((cnx + 7)/8, (cny + 7)/8, (cnz + 7)/8);
    restrict_kernel_3d<<<cgrid, cblock, 0, stream>>>(
        L.g.x, L.g.b, L.g.solid, coarse.g.b, coarse.g.solid,
        nx, ny, nz, L.g.pitch, coarse.g.pitch,
        L.g.idx2, L.g.idy2, L.g.idz2, L.g.diag);

    // Zero coarse x & recurse
    int Nc = (cnx+2)*(cny+2)*(cnz+2);
    zero_kernel_3d<<<(Nc+255)/256, 256, 0, stream>>>(coarse.g.x, Nc);
    vCycle3D(levels, level + 1, nlevels, stream);

    // Prolongate
    dim3 fblock(8, 8, 8);
    dim3 fgrid((nx + 7)/8, (ny + 7)/8, (nz + 7)/8);
    prolong_kernel_3d<<<fgrid, fblock, 0, stream>>>(
        L.g.x, coarse.g.x, L.g.solid, nx, ny, nz, L.g.pitch, coarse.g.pitch);

    // Post-smooth
    rbgs_sweep_3d(L, stream);
}

// ── CudaUAAMGPreconditioner3D ──

void CudaUAAMGPreconditioner3D::build(const CudaGrid3D& fine) {
    if (cached_nx_ == fine.nx && cached_ny_ == fine.ny && cached_nz_ == fine.nz) return;
    destroy();
    int nx = fine.nx, ny = fine.ny, nz = fine.nz;
    double dx = fine.dx, dy = fine.dy, dz = fine.dz;
    while (nx >= 2 && ny >= 2 && nz >= 2) {
        Level L;
        L.g.allocate(nx, ny, nz, dx, dy, dz);
        L.stride = nx + 2;
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4) break;
        nx /= 2; ny /= 2; nz /= 2; dx *= 2.0; dy *= 2.0; dz *= 2.0;
    }
    cached_nx_ = fine.nx; cached_ny_ = fine.ny; cached_nz_ = fine.nz;
}

void CudaUAAMGPreconditioner3D::apply(const CudaGrid3D& fine, const double* r, double* z) {
    build(fine);
    int nl = (int)levels_.size();
    cudaStream_t stream = 0;
    int N0 = (fine.nx+2)*(fine.ny+2)*(fine.nz+2);

    // Copy solid to finest
    cudaMemcpy(levels_[0].g.solid, fine.solid, N0 * sizeof(bool), cudaMemcpyDeviceToDevice);

    // Propagate solid down
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l-1];
        auto& cL = levels_[l];
        dim3 block(8, 8, 8);
        dim3 grid((cL.g.nx + 7)/8, (cL.g.ny + 7)/8, (cL.g.nz + 7)/8);
        restrict_solid_kernel_3d<<<grid, block, 0, stream>>>(
            fL.g.solid, cL.g.solid, fL.g.nx, fL.g.ny, fL.g.nz, fL.g.pitch, cL.g.pitch);
    }

    // Copy r → finest b
    copy_kernel_3d<<<(N0+255)/256, 256, 0, stream>>>(levels_[0].g.b, r, N0);

    // Zero all pressures
    for (int l = 0; l < nl; l++) {
        int N = (levels_[l].g.nx+2)*(levels_[l].g.ny+2)*(levels_[l].g.nz+2);
        zero_kernel_3d<<<(N+255)/256, 256, 0, stream>>>(levels_[l].g.x, N);
    }

    vCycle3D(levels_.data(), 0, nl, stream);
    cudaDeviceSynchronize();

    // Copy finest correction → z
    copy_kernel_3d<<<(N0+255)/256, 256, 0, stream>>>(z, levels_[0].g.x, N0);
    cudaDeviceSynchronize();
}

void CudaUAAMGPreconditioner3D::destroy() {
    for (auto& L : levels_) L.g.free();
    levels_.clear();
    cached_nx_ = cached_ny_ = cached_nz_ = -1;
}

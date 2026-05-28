/**
 * @file cuda_uaamg_preconditioner.cu
 * @brief CUDA 2D UAAMG V-cycle — matrix-free Galerkin (Algorithm 3, Section 5.1).
 *
 * Sun et al., "Leapfrog Flow Maps for Real-Time Fluid Simulation", SIGGRAPH 2025.
 *
 * Paper-faithful:
 *   - Constant prolongation with ×2 correction scaling (Eq. 11);
 *   - 4-to-1 restriction R = Pᵀ (sum over 2×2 children);
 *   - Galerkin coarse operator A_{l+1} = R_l A_l P_l (Eq. 12), kept matrix-free:
 *     coarse +x coupling = sum of the 2 fine +x couplings across the shared edge;
 *     coarse diagonal = sum of the 4 coarse couplings (Neumann zero row sum);
 *   - RBGS smoother, symmetric V(1,1) (forward pre-smooth + reverse post-smooth)
 *     so the preconditioner is SPD for CG.
 *
 * Each level stores stencil channels (diag, cx, cy); cx[c] couples c to its +x
 * neighbour (A[c,c+ex] = -cx[c]); the −x coupling is cx[c-ex].
 */
#include "solver/cuda/cuda_uaamg_preconditioner.h"

__device__ inline int dev_idx(int i, int j, int stride) { return i + j * stride; }

// ── Finest-level stencil from the solid mask ──
__global__ void setup_fine_coeffs_kernel(
    const bool *solid, double *cx, double *cy,
    int nx, int ny, int stride, double idx2, double idy2)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (i>nx || j>ny) return;
    int id = dev_idx(i,j,stride);
    if (solid[id]) { cx[id]=cy[id]=0.0; return; }
    cx[id] = (i<nx && !solid[dev_idx(i+1,j,stride)]) ? idx2 : 0.0;
    cy[id] = (j<ny && !solid[dev_idx(i,j+1,stride)]) ? idy2 : 0.0;
}

// diag[c] = sum of the 4 active couplings (fine or coarse level)
__global__ void setup_diag_kernel(
    const bool *solid, const double *cx, const double *cy, double *diag,
    int nx, int ny, int stride)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (i>nx || j>ny) return;
    int id = dev_idx(i,j,stride);
    if (solid[id]) { diag[id]=0.0; return; }
    diag[id] = cx[id] + cx[dev_idx(i-1,j,stride)] + cy[id] + cy[dev_idx(i,j-1,stride)];
}

// ── Galerkin coarse couplings: sum the 2 fine couplings on each shared edge ──
__global__ void galerkin_coeffs_kernel(
    const double *fcx, const double *fcy,
    const bool *csolid, double *ccx, double *ccy,
    int fstride, int cnx, int cny, int cstride)
{
    int ic = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (ic>cnx || jc>cny) return;
    int cid = dev_idx(ic,jc,cstride);
    if (csolid[cid]) { ccx[cid]=ccy[cid]=0.0; return; }
    int i_f=2*ic-1, j_f=2*jc-1;
    double sx=0, sy=0;
    for (int dj=0; dj<2; dj++) sx += fcx[dev_idx(i_f+1, j_f+dj, fstride)];  // +x edge = fine i_f+1
    for (int di=0; di<2; di++) sy += fcy[dev_idx(i_f+di, j_f+1, fstride)];
    ccx[cid]=sx; ccy[cid]=sy;
}

// ── RBGS sweep (one parity) using stored coefficients ──
__global__ void rbgs_coeff_kernel(
    double *x, const double *b, const bool *solid,
    const double *diag, const double *cx, const double *cy,
    int nx, int ny, int stride, int parity)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (i>nx || j>ny) return;
    if (((i+j)&1) != parity) return;
    int id = dev_idx(i,j,stride);
    if (solid[id] || diag[id] < 1e-30) return;
    int im=dev_idx(i-1,j,stride), jm=dev_idx(i,j-1,stride);
    double nb = cx[id]*x[dev_idx(i+1,j,stride)] + cx[im]*x[im]
              + cy[id]*x[dev_idx(i,j+1,stride)] + cy[jm]*x[jm];
    x[id] = (b[id] + nb) / diag[id];
}

// ── Residual restriction R = Pᵀ (sum over 2×2 children) ──
__global__ void restrict_coeff_kernel(
    const double *xf, const double *bf, const bool *fsolid,
    const double *fdiag, const double *fcx, const double *fcy,
    double *bc, const bool *csolid,
    int fnx, int fny, int fstride, int cnx, int cny, int cstride)
{
    int ic = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (ic>cnx || jc>cny) return;
    int cid = dev_idx(ic,jc,cstride);
    if (csolid[cid]) return;
    int i_f=2*ic-1, j_f=2*jc-1;
    double sum=0;
    for (int di=0;di<2;di++) for(int dj=0;dj<2;dj++) {
        int fi=i_f+di, fj=j_f+dj, fid=dev_idx(fi,fj,fstride);
        if (fsolid[fid]) continue;
        int im=dev_idx(fi-1,fj,fstride), jm=dev_idx(fi,fj-1,fstride);
        double Ax = fdiag[fid]*xf[fid]
                  - fcx[fid]*xf[dev_idx(fi+1,fj,fstride)] - fcx[im]*xf[im]
                  - fcy[fid]*xf[dev_idx(fi,fj+1,fstride)] - fcy[jm]*xf[jm];
        sum += bf[fid] - Ax;
    }
    bc[cid] = sum;
}

// ── Prolongation: constant injection with ×2 scaling (paper Eq. 11) ──
__global__ void prolong_kernel(
    double *x_fine, const double *x_coarse, const bool *solid_fine,
    int fnx, int fny, int fstride, int cstride)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (i>fnx || j>fny) return;
    int fid = dev_idx(i,j,fstride);
    if (solid_fine[fid]) return;
    int ic=(i+1)/2, jc=(j+1)/2;
    x_fine[fid] += 2.0 * x_coarse[dev_idx(ic,jc,cstride)];
}

// ── Restrict solid mask (coarse solid if ≥2 of 4 fine solid) ──
__global__ void restrict_solid_kernel(
    const bool *solid_fine, bool *solid_coarse,
    int fnx, int fny, int fstride, int cstride)
{
    int ic = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int cnx = fnx/2, cny = fny/2;
    if (ic>cnx || jc>cny) return;
    int i_f=2*ic-1, j_f=2*jc-1, sc=0;
    for (int di=0;di<2;di++) for (int dj=0;dj<2;dj++)
        if (solid_fine[dev_idx(i_f+di,j_f+dj,fstride)]) sc++;
    solid_coarse[dev_idx(ic,jc,cstride)] = (sc >= 2);
}

__global__ void zero_kernel(double *a, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) a[i] = 0.0;
}
__global__ void copy_kernel(double *dst, const double *src, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ── one RBGS sweep (forward = odd,even ; reverse = even,odd) ──
static void rbgs_sweep(CudaUAAMGPreconditioner::Level& L, cudaStream_t stream, bool reverse) {
    int nx = L.g.nx, ny = L.g.ny;
    dim3 block(16,16), grid((nx+15)/16, (ny+15)/16);
    int p0 = reverse ? 0 : 1;
    rbgs_coeff_kernel<<<grid,block,0,stream>>>(L.g.x,L.g.b,L.g.solid,L.diag,L.cx,L.cy,nx,ny,L.g.pitch,p0);
    rbgs_coeff_kernel<<<grid,block,0,stream>>>(L.g.x,L.g.b,L.g.solid,L.diag,L.cx,L.cy,nx,ny,L.g.pitch,1-p0);
}

// ── Symmetric V(1,1) ──
static void vCycle(CudaUAAMGPreconditioner::Level* levels, int level, int nlevels, cudaStream_t stream) {
    auto& L = levels[level];
    int nx = L.g.nx, ny = L.g.ny;
    if (level == nlevels - 1) {
        for (int s=0;s<10;s++) { rbgs_sweep(L,stream,false); rbgs_sweep(L,stream,true); }
        return;
    }
    rbgs_sweep(L, stream, false);                        // pre-smooth (forward)

    auto& coarse = levels[level + 1];
    int cnx = coarse.g.nx, cny = coarse.g.ny;
    dim3 cblock(16,16), cgrid((cnx+15)/16, (cny+15)/16);
    restrict_coeff_kernel<<<cgrid,cblock,0,stream>>>(
        L.g.x, L.g.b, L.g.solid, L.diag, L.cx, L.cy,
        coarse.g.b, coarse.g.solid, nx, ny, L.g.pitch, cnx, cny, coarse.g.pitch);

    int Nc = (cnx+2)*(cny+2);
    zero_kernel<<<(Nc+255)/256, 256, 0, stream>>>(coarse.g.x, Nc);
    vCycle(levels, level + 1, nlevels, stream);

    dim3 fblock(16,16), fgrid((nx+15)/16, (ny+15)/16);
    prolong_kernel<<<fgrid,fblock,0,stream>>>(
        L.g.x, coarse.g.x, L.g.solid, nx, ny, L.g.pitch, coarse.g.pitch);

    rbgs_sweep(L, stream, true);                         // post-smooth (reverse → symmetric)
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
        int N = (nx+2)*(ny+2);
        cudaMalloc(&L.diag, N*sizeof(double)); cudaMemset(L.diag,0,N*sizeof(double));
        cudaMalloc(&L.cx,   N*sizeof(double)); cudaMemset(L.cx,  0,N*sizeof(double));
        cudaMalloc(&L.cy,   N*sizeof(double)); cudaMemset(L.cy,  0,N*sizeof(double));
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

    cudaMemcpy(levels_[0].g.solid, fine.solid, N0 * sizeof(bool), cudaMemcpyDeviceToDevice);
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l-1]; auto& cL = levels_[l];
        dim3 block(16,16), grid((cL.g.nx+15)/16, (cL.g.ny+15)/16);
        restrict_solid_kernel<<<grid,block,0,stream>>>(
            fL.g.solid, cL.g.solid, fL.g.nx, fL.g.ny, fL.g.pitch, cL.g.pitch);
    }

    // Stencil coefficients: finest from solid, coarse via Galerkin.
    {
        auto& L = levels_[0]; int nx=L.g.nx, ny=L.g.ny;
        dim3 block(16,16), grid((nx+15)/16, (ny+15)/16);
        setup_fine_coeffs_kernel<<<grid,block,0,stream>>>(
            L.g.solid, L.cx, L.cy, nx, ny, L.g.pitch, L.g.idx2, L.g.idy2);
        setup_diag_kernel<<<grid,block,0,stream>>>(
            L.g.solid, L.cx, L.cy, L.diag, nx, ny, L.g.pitch);
    }
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l-1]; auto& cL = levels_[l];
        int cnx=cL.g.nx, cny=cL.g.ny;
        dim3 block(16,16), grid((cnx+15)/16, (cny+15)/16);
        galerkin_coeffs_kernel<<<grid,block,0,stream>>>(
            fL.cx, fL.cy, cL.g.solid, cL.cx, cL.cy, fL.g.pitch, cnx, cny, cL.g.pitch);
        setup_diag_kernel<<<grid,block,0,stream>>>(
            cL.g.solid, cL.cx, cL.cy, cL.diag, cnx, cny, cL.g.pitch);
    }

    copy_kernel<<<(N0+255)/256, 256, 0, stream>>>(levels_[0].g.b, r, N0);
    for (int l = 0; l < nl; l++) {
        int N = (levels_[l].g.nx+2)*(levels_[l].g.ny+2);
        zero_kernel<<<(N+255)/256, 256, 0, stream>>>(levels_[l].g.x, N);
    }
    vCycle(levels_.data(), 0, nl, stream);
    cudaDeviceSynchronize();

    copy_kernel<<<(N0+255)/256, 256, 0, stream>>>(z, levels_[0].g.x, N0);
    cudaDeviceSynchronize();
}

void CudaUAAMGPreconditioner::destroy() {
    for (auto& L : levels_) {
        L.g.free();
        if (L.diag) cudaFree(L.diag);
        if (L.cx)   cudaFree(L.cx);
        if (L.cy)   cudaFree(L.cy);
        L.diag=L.cx=L.cy=nullptr;
    }
    levels_.clear();
    cached_nx_ = cached_ny_ = -1;
}

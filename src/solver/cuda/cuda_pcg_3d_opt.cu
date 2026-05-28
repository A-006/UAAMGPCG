/**
 * @file cuda_pcg_3d_opt.cu
 * @brief Optimized PCG with fused dot products + shared-memory tiling.
 *
 * Step 1: dot(p,Ap) fused into matvec kernel (saves 1 launch/iter)
 * Step 2: dot(r,r) fused into axpy kernel (saves 1 launch/iter)
 * Step 3: Shared-memory tiling for matvec (8^3 tile + halo → fewer global reads)
 *
 * PCG loop kernel budget: 12 → 8 launches (33% reduction)
 */
#include "solver/cuda/cuda_pcg_3d.h"
#include <cstdio>
#include <cmath>

__device__ inline int opti_idx(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

// ── Shared-memory reduction helper ──
template<int BS>
__device__ double block_reduce(double val, double *smem, int tid) {
    smem[tid] = val;
    __syncthreads();
    for (int s = BS/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

// ═══════════════════════════════════════════════════════════════
//  Optimized Matvec: Shared-memory tiling + fused dot(p,Ap)
//
//  Load 8^3 tile + 1-cell halo into shared memory.
//  Each cell's 6 neighbors read from shared memory (not global).
//  Also computes dot(p,Ap) partial sum per block.
// ═══════════════════════════════════════════════════════════════
#define T 8          // tile dim
#define TH (T+2)     // with halo

__global__ void matvec_tiled_dot_kernel(
    const double *p, double *Ap, const bool *solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag,
    double *dot_buf)
{
    __shared__ double sp[TH][TH][TH];  // p values with halo
    __shared__ bool   ss[TH][TH][TH];  // solid mask with halo
    __shared__ double sdot[TH*TH*TH];  // for dot reduction

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gi = blockIdx.x * T + tx + 1;  // global i (1-based interior)
    int gj = blockIdx.y * T + ty + 1;
    int gk = blockIdx.z * T + tz + 1;
    int li = tx + 1, lj = ty + 1, lk = tz + 1;  // local pos in shared mem

    // ── Load interior cell ──
    bool valid = (gi <= nx && gj <= ny && gk <= nz);
    if (valid) {
        int gid = opti_idx(gi, gj, gk, pitch, ny);
        sp[li][lj][lk] = p[gid];
        ss[li][lj][lk] = solid[gid];
    } else {
        ss[li][lj][lk] = true;
        sp[li][lj][lk] = 0.0;
    }

    // ── Load x-face halos (threads at tx=0 or tx=T-1) ──
    if (tx == 0) {
        int gl = gi - 1;
        if (gl >= 1 && gj <= ny && gk <= nz) {
            int gid = opti_idx(gl, gj, gk, pitch, ny);
            sp[0][lj][lk] = p[gid]; ss[0][lj][lk] = solid[gid];
        } else { ss[0][lj][lk] = true; sp[0][lj][lk] = 0.0; }
    }
    if (tx == T - 1) {
        int gr = gi + 1;
        if (gr <= nx && gj <= ny && gk <= nz) {
            int gid = opti_idx(gr, gj, gk, pitch, ny);
            sp[T+1][lj][lk] = p[gid]; ss[T+1][lj][lk] = solid[gid];
        } else { ss[T+1][lj][lk] = true; sp[T+1][lj][lk] = 0.0; }
    }

    // ── Load y-face halos ──
    if (ty == 0) {
        int gb = gj - 1;
        if (gi <= nx && gb >= 1 && gk <= nz) {
            int gid = opti_idx(gi, gb, gk, pitch, ny);
            sp[li][0][lk] = p[gid]; ss[li][0][lk] = solid[gid];
        } else { ss[li][0][lk] = true; sp[li][0][lk] = 0.0; }
    }
    if (ty == T - 1) {
        int gt = gj + 1;
        if (gi <= nx && gt <= ny && gk <= nz) {
            int gid = opti_idx(gi, gt, gk, pitch, ny);
            sp[li][T+1][lk] = p[gid]; ss[li][T+1][lk] = solid[gid];
        } else { ss[li][T+1][lk] = true; sp[li][T+1][lk] = 0.0; }
    }

    // ── Load z-face halos ──
    if (tz == 0) {
        int gf = gk - 1;
        if (gi <= nx && gj <= ny && gf >= 1) {
            int gid = opti_idx(gi, gj, gf, pitch, ny);
            sp[li][lj][0] = p[gid]; ss[li][lj][0] = solid[gid];
        } else { ss[li][lj][0] = true; sp[li][lj][0] = 0.0; }
    }
    if (tz == T - 1) {
        int gk2 = gk + 1;
        if (gi <= nx && gj <= ny && gk2 <= nz) {
            int gid = opti_idx(gi, gj, gk2, pitch, ny);
            sp[li][lj][T+1] = p[gid]; ss[li][lj][T+1] = solid[gid];
        } else { ss[li][lj][T+1] = true; sp[li][lj][T+1] = 0.0; }
    }

    __syncthreads(); // all halos loaded

    // ── Compute matvec + partial dot product ──
    double my_dot = 0.0;
    double Ap_val = 0.0;

    if (valid && !ss[li][lj][lk]) {
        double pC = sp[li][lj][lk];
        double pL = ss[li-1][lj][lk] ? pC : sp[li-1][lj][lk];
        double pR = ss[li+1][lj][lk] ? pC : sp[li+1][lj][lk];
        double pB = ss[li][lj-1][lk] ? pC : sp[li][lj-1][lk];
        double pT = ss[li][lj+1][lk] ? pC : sp[li][lj+1][lk];
        double pF = ss[li][lj][lk-1] ? pC : sp[li][lj][lk-1];
        double pK = ss[li][lj][lk+1] ? pC : sp[li][lj][lk+1];
        Ap_val = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2 - (pF+pK)*idz2;

        int gid = opti_idx(gi, gj, gk, pitch, ny);
        Ap[gid] = Ap_val;
        my_dot = pC * Ap_val;
    }

    // ── Block-level dot reduction ──
    int tid = tx + ty * T + tz * (T*T);
    sdot[tid] = my_dot;
    __syncthreads();
    for (int s = (T*T*T)/2; s > 0; s >>= 1) {
        if (tid < s) sdot[tid] += sdot[tid + s];
        __syncthreads();
    }

    // Write partial dot to buffer
    if (tid == 0) {
        int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        dot_buf[bid] = sdot[0];
    }
}

// ═══════════════════════════════════════════════════════════════
//  Optimized AXPY + fused dot(r,r): r += a*Ap, partial dot(r²)
// ═══════════════════════════════════════════════════════════════
__global__ void axpy_dot_kernel(
    double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int nz, int pitch,
    double *dot_buf)
{
    __shared__ double sdot[T*T*T];

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    double my_dot = 0.0;
    if (i <= nx && j <= ny && k <= nz) {
        int id = opti_idx(i, j, k, pitch, ny);
        if (!solid[id]) {
            y[id] += a * x[id];
            my_dot = y[id] * y[id];
        }
    }

    int tid = threadIdx.x + threadIdx.y * T + threadIdx.z * (T*T);
    sdot[tid] = my_dot;
    __syncthreads();
    for (int s = (T*T*T)/2; s > 0; s >>= 1) {
        if (tid < s) sdot[tid] += sdot[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        dot_buf[bid] = sdot[0];
    }
}

// ── Standalone kernels (used where fusion not applicable) ──

__global__ void dot_kernel_1d(
    const double *a, const double *b, const bool *solid, int N, double *part)
{
    __shared__ double s[256];
    int tid = threadIdx.x; double sum = 0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x)
        if (!solid[k]) sum += a[k] * b[k];
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void sum_interior_kernel(
    const double *v, const bool *solid, int nx, int ny, int nz, int pitch, double *part)
{
    __shared__ double s[256]; int tid = threadIdx.x; double sum = 0;
    for (int lin = blockIdx.x * blockDim.x + tid; lin < nx*ny*nz; lin += blockDim.x * gridDim.x) {
        int i = (lin % nx) + 1, j = ((lin / nx) % ny) + 1, k = (lin / (nx * ny)) + 1;
        int id = opti_idx(i, j, k, pitch, ny);
        if (!solid[id]) sum += v[id];
    }
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void count_interior_kernel(
    const bool *solid, int nx, int ny, int nz, int pitch, int *part)
{
    __shared__ int s[256]; int tid = threadIdx.x; int sum = 0;
    for (int lin = blockIdx.x * blockDim.x + tid; lin < nx*ny*nz; lin += blockDim.x * gridDim.x) {
        int i = (lin % nx) + 1, j = ((lin / nx) % ny) + 1, k = (lin / (nx * ny)) + 1;
        if (!solid[opti_idx(i, j, k, pitch, ny)]) sum++;
    }
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void submean_kernel(double *v, double m, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x+1, j=blockIdx.y*blockDim.y+threadIdx.y+1, k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz) return;
    int id=opti_idx(i,j,k,pitch,ny);
    if(!solid[id]) v[id]-=m;
}

__global__ void negate_kernel(double *v, const bool *solid, int nx, int ny, int nz, int pitch)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x+1, j=blockIdx.y*blockDim.y+threadIdx.y+1, k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz) return;
    int id=opti_idx(i,j,k,pitch,ny);
    if(!solid[id]) v[id]=-v[id];
}

__global__ void axpy_kernel(double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x+1, j=blockIdx.y*blockDim.y+threadIdx.y+1, k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz) return;
    int id=opti_idx(i,j,k,pitch,ny);
    if(!solid[id]) y[id]+=a*x[id];
}

// ── GPU-side reduction: sum partials to scalar ──
__global__ void reduce_final_kernel(const double *part, int n, double *result) {
    __shared__ double s[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < n; i += 256) sum += part[i];
    s[tid] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    if (tid == 0) *result = s[0];
}
static double gpu_sum(const double *d_part, int n, double *d_result) {
    reduce_final_kernel<<<1, 256>>>(d_part, n, d_result);
    double val; cudaMemcpy(&val, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return val;
}

// ── Host reduction ──
static double host_sum(const double *d, int n) {
    std::vector<double> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost);
    double t = 0; for (double v : h) t += v; return t;
}
static int host_sum_int(const int *d, int n) {
    std::vector<int> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
    int t = 0; for (int v : h) t += v; return t;
}
static double gpu_mean(const double *v, const bool *solid,
    int nx, int ny, int nz, int pitch, double *part, int *cnt, int nb)
{
    sum_interior_kernel<<<nb,256>>>(v,solid,nx,ny,nz,pitch,part); cudaDeviceSynchronize();
    double s=gpu_sum(part,nb,part);
    count_interior_kernel<<<nb,256>>>(solid,nx,ny,nz,pitch,cnt); cudaDeviceSynchronize();
    int c=host_sum_int(cnt,nb);
    return c>0?s/c:0.0;
}

// ── Device-resident scalar ops (no host sync; for solve_device) ──
__global__ void axpy_dev_k(double *y, const double *x, const double *a, const bool *s,
    int nx,int ny,int nz,int pitch){
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz)return; int id=opti_idx(i,j,k,pitch,ny); if(!s[id]) y[id]+=(*a)*x[id];
}
__global__ void submean_dev_k(double *v, const double *m, const bool *s,
    int nx,int ny,int nz,int pitch){
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz)return; int id=opti_idx(i,j,k,pitch,ny); if(!s[id]) v[id]-=(*m);
}
__global__ void pupd_dev_k(double *p, const double *z, const double *b, const bool *s,
    int nx,int ny,int nz,int pitch){    // p = z + beta*p  (search-direction update)
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz)return; int id=opti_idx(i,j,k,pitch,ny); if(!s[id]) p[id]=z[id]+(*b)*p[id];
}
__global__ void sc_div_k   (double*o,const double*n,const double*d){ *o = *n / *d; }
__global__ void sc_negdiv_k(double*o,const double*n,const double*d){ *o = -(*n) / *d; }
__global__ void sc_meandiv_k(double*o,const double*s,double cnt){ *o = (cnt>0)? *s/cnt : 0.0; }
__global__ void sc_copy_k  (double*o,const double*i){ *o = *i; }

// ── Mixed precision: FP64↔FP32 casts + FP32 preconditioner apply ──
__global__ void cast_d2f_k(const double* d, float* f, int N){ int i=blockIdx.x*256+threadIdx.x; if(i<N) f[i]=(float)d[i]; }
__global__ void cast_f2d_k(const float* f, double* d, int N){ int i=blockIdx.x*256+threadIdx.x; if(i<N) d[i]=(double)f[i]; }

void CudaPCG3D::mixed_apply(int N, const double* dr, double* dz) {
    int nb=(N+255)/256;
    cast_d2f_k<<<nb,256>>>(dr, d_rf, N);          // r (FP64) → r (FP32)
    precond_f_->vcycle_apply(gf_, d_rf, d_zf);    // FP32 Galerkin V-cycle
    cast_f2d_k<<<nb,256>>>(d_zf, dz, N);          // z (FP32) → z (FP64)
}

void CudaPCG3D::solve_mixed(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol) {
    mixed_ = true;
    solve_optimized(g, p, rhs, max_iter, tol);
    mixed_ = false;
}

// ═══════════════════════════════════════════════════════════════
//  Optimized CudaPCG3D::solve_optimized
// ═══════════════════════════════════════════════════════════════
void CudaPCG3D::solve_optimized(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol) {
    int nx=g.nx, ny=g.ny, nz=g.nz, pitch=g.pitch, N=(nx+2)*(ny+2)*(nz+2);
    ensure_buffers(N);
    dim3 block3d(T,T,T), grid3d((nx+T-1)/T,(ny+T-1)/T,(nz+T-1)/T);
    int n3d = grid3d.x * grid3d.y * grid3d.z; // blocks for fused kernels
    int nb1d = dot_buf_size_;

    if (mixed_) {               // FP32 preconditioner: float grid + setup once
        if (gf_N_ < N) {
            if (gf_N_ > 0) { gf_.free(); if(d_rf)cudaFree(d_rf); if(d_zf)cudaFree(d_zf); }
            gf_.allocate(nx,ny,nz,(float)g.dx,(float)g.dy,(float)g.dz);
            cudaMalloc(&d_rf, N*sizeof(float)); cudaMalloc(&d_zf, N*sizeof(float));
            gf_N_ = N;
        }
        cudaMemcpy(gf_.solid, g.solid, N*sizeof(bool), cudaMemcpyDeviceToDevice);
        precond_f_->setupLevels(gf_);
    } else {
        precond_->setupLevels(g);   // solid + Galerkin coeffs + §5.4 trimming — once per solve
    }

    // Init
    cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N*sizeof(double));

    double mr = gpu_mean(d_r, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
    submean_kernel<<<grid3d,block3d>>>(d_r, mr, g.solid, nx, ny, nz, pitch);
    negate_kernel<<<grid3d,block3d>>>(d_r, g.solid, nx, ny, nz, pitch);

    // Preconditioner z = M⁻¹ r (FP32 in mixed mode, FP64 otherwise).
    if (mixed_) mixed_apply(N, d_r, d_z); else precond_->vcycle_apply(g, d_r, d_z);

    double mz = gpu_mean(d_z, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
    submean_kernel<<<grid3d,block3d>>>(d_z, mz, g.solid, nx, ny, nz, pitch);
    cudaMemcpy(d_p, d_z, N*sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r,z)
    dot_kernel_1d<<<nb1d,256>>>(d_r, d_z, g.solid, N, d_dot_buf);
    cudaDeviceSynchronize();
    double rsold = gpu_sum(d_dot_buf, nb1d, d_scalar);
    if (rsold < 1e-30) { cudaMemset(p, 0, N*sizeof(double)); return; }

    // Compute initial absolute L2(r) for relative-residual stopping (||r||/||r0|| < tol).
    dot_kernel_1d<<<nb1d,256>>>(d_r, d_r, g.solid, N, d_dot_buf);
    cudaDeviceSynchronize();
    double r0_sq = gpu_sum(d_dot_buf, nb1d, d_scalar);
    double tol_abs_sq = (r0_sq > 0) ? r0_sq * tol * tol : tol * tol;

    last_iters = max_iter;
    last_rel_res = 1.0;
    static bool dbg = std::getenv("PCG_DBG") != nullptr;
    if (dbg) std::printf("[PCG] r0_sq=%.3e tol_abs_sq=%.3e\n", r0_sq, tol_abs_sq);
    for (int k = 0; k < max_iter; k++) {
        // ── FUSED: matvec(tiled) + dot(p,Ap) ──
        matvec_tiled_dot_kernel<<<grid3d,block3d>>>(
            d_p, d_Ap, g.solid, nx, ny, nz, pitch,
            g.idx2, g.idy2, g.idz2, g.diag,
            d_dot_buf);
        cudaDeviceSynchronize();
        double pAp = gpu_sum(d_dot_buf, n3d, d_scalar);
        if (pAp < 1e-15) break;

        double alpha = rsold / pAp;

        // axpy: x += alpha*p (no dot needed)
        axpy_kernel<<<grid3d,block3d>>>(p, d_p, alpha, g.solid, nx, ny, nz, pitch);

        // ── FUSED: axpy(r, -alpha*Ap) + dot(r,r) ──
        axpy_dot_kernel<<<grid3d,block3d>>>(
            d_r, d_Ap, -alpha, g.solid, nx, ny, nz, pitch,
            d_dot_buf);
        cudaDeviceSynchronize();
        double rsnew = gpu_sum(d_dot_buf, n3d, d_scalar);

        // Relative residual: ||r||/||r0|| < tol  ⇔  rsnew < tol² * r0_sq
        if (dbg && (k<5 || k%5==0)) std::printf("[PCG]  k=%3d rsnew=%.3e rel=%.3e\n", k, rsnew, std::sqrt(rsnew/std::max(r0_sq,1e-30)));
        if (rsnew < tol_abs_sq) {
            last_iters = k + 1;
            last_rel_res = std::sqrt(rsnew / (r0_sq > 0 ? r0_sq : 1.0));
            break;
        }

        if (mixed_) mixed_apply(N, d_r, d_z); else precond_->vcycle_apply(g, d_r, d_z);

        double mz2 = gpu_mean(d_z, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
        submean_kernel<<<grid3d,block3d>>>(d_z, mz2, g.solid, nx, ny, nz, pitch);
        cudaDeviceSynchronize();

        dot_kernel_1d<<<nb1d,256>>>(d_r, d_z, g.solid, N, d_dot_buf);
        cudaDeviceSynchronize();
        double rz = gpu_sum(d_dot_buf, nb1d, d_scalar);

        double beta = rz / rsold;
        rsold = rz;

        cudaMemcpy(d_Ap, d_p, N*sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_z, N*sizeof(double), cudaMemcpyDeviceToDevice);
        axpy_kernel<<<grid3d,block3d>>>(d_p, d_Ap, beta, g.solid, nx, ny, nz, pitch);
        cudaDeviceSynchronize();
    }
}

// ═══════════════════════════════════════════════════════════════
//  Fully device-resident PCG (no per-iteration host sync / cudaMemcpy).
//  All scalars (rsold, pAp, rsnew, rz, alpha, beta, mean) live on device;
//  alpha/beta and the mean-projection are computed by 1-thread kernels; the
//  search-direction update is in place. Fixed iteration count (the paper does
//  the same to avoid interrupting the stream for residual checks). Returns the
//  final ||r||² in last_rel_res-friendly form via one end-of-solve copy.
// ═══════════════════════════════════════════════════════════════
void CudaPCG3D::solve_device(CudaGrid3D& g, double* p, double* rhs, int iters, double /*tol*/) {
    int nx=g.nx, ny=g.ny, nz=g.nz, pitch=g.pitch, N=(nx+2)*(ny+2)*(nz+2);
    ensure_buffers(N);
    dim3 blk(T,T,T), grd((nx+T-1)/T,(ny+T-1)/T,(nz+T-1)/T);
    int n3d = grd.x*grd.y*grd.z, nb1d = dot_buf_size_;

    precond_->setupLevels(g);

    double* d; cudaMalloc(&d, 8*sizeof(double));   // 0 rsold,1 pAp,2 rsnew,3 rz,4 alpha,5 negalpha,6 beta,7 mean
    // fluid-cell count once (constant during the solve)
    count_interior_kernel<<<nb1d,256>>>(g.solid,nx,ny,nz,pitch,d_count_buf);
    cudaDeviceSynchronize();
    double dcnt = (double)host_sum_int(d_count_buf, nb1d);

    auto dmean = [&](double* v){    // *d[7] = mean(v over fluid); v -= mean   (all device)
        sum_interior_kernel<<<nb1d,256>>>(v,g.solid,nx,ny,nz,pitch,d_dot_buf);
        reduce_final_kernel<<<1,256>>>(d_dot_buf,nb1d,d+7);
        sc_meandiv_k<<<1,1>>>(d+7,d+7,dcnt);
        submean_dev_k<<<grd,blk>>>(v,d+7,g.solid,nx,ny,nz,pitch);
    };

    cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N*sizeof(double));
    dmean(d_r); negate_kernel<<<grd,blk>>>(d_r,g.solid,nx,ny,nz,pitch);  // r = -(rhs-mean)
    precond_->vcycle_apply(g, d_r, d_z); dmean(d_z);
    cudaMemcpy(d_p, d_z, N*sizeof(double), cudaMemcpyDeviceToDevice);
    dot_kernel_1d<<<nb1d,256>>>(d_r,d_z,g.solid,N,d_dot_buf);
    reduce_final_kernel<<<1,256>>>(d_dot_buf,nb1d,d+0);                  // rsold = (r,z)

    for (int k=0;k<iters;k++){
        matvec_tiled_dot_kernel<<<grd,blk>>>(d_p,d_Ap,g.solid,nx,ny,nz,pitch,g.idx2,g.idy2,g.idz2,g.diag,d_dot_buf);
        reduce_final_kernel<<<1,256>>>(d_dot_buf,n3d,d+1);               // pAp
        sc_div_k   <<<1,1>>>(d+4,d+0,d+1);                              // alpha = rsold/pAp
        sc_negdiv_k<<<1,1>>>(d+5,d+0,d+1);                              // -alpha
        axpy_dev_k<<<grd,blk>>>(p,   d_p, d+4, g.solid,nx,ny,nz,pitch); // x += alpha p
        axpy_dev_k<<<grd,blk>>>(d_r, d_Ap,d+5, g.solid,nx,ny,nz,pitch); // r -= alpha Ap
        precond_->vcycle_apply(g, d_r, d_z); dmean(d_z);
        dot_kernel_1d<<<nb1d,256>>>(d_r,d_z,g.solid,N,d_dot_buf);
        reduce_final_kernel<<<1,256>>>(d_dot_buf,nb1d,d+3);             // rz
        sc_div_k <<<1,1>>>(d+6,d+3,d+0);                               // beta = rz/rsold
        sc_copy_k<<<1,1>>>(d+0,d+3);                                   // rsold = rz
        pupd_dev_k<<<grd,blk>>>(d_p,d_z,d+6,g.solid,nx,ny,nz,pitch);    // p = z + beta p
    }
    // one host sync at the very end (final residual for reporting)
    dot_kernel_1d<<<nb1d,256>>>(d_r,d_r,g.solid,N,d_dot_buf);
    double rsn = gpu_sum(d_dot_buf,nb1d,d_scalar);
    last_iters = iters; last_rel_res = std::sqrt(rsn);
    cudaFree(d);
}

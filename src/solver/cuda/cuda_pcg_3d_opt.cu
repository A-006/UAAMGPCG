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
    double s=host_sum(part,nb);
    count_interior_kernel<<<nb,256>>>(solid,nx,ny,nz,pitch,cnt); cudaDeviceSynchronize();
    int c=host_sum_int(cnt,nb);
    return c>0?s/c:0.0;
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

    precond_->build(g);

    // Init
    cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N*sizeof(double));

    double mr = gpu_mean(d_r, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
    submean_kernel<<<grid3d,block3d>>>(d_r, mr, g.solid, nx, ny, nz, pitch);
    negate_kernel<<<grid3d,block3d>>>(d_r, g.solid, nx, ny, nz, pitch);

    precond_->apply_optimized(g, d_r, d_z);

    double mz = gpu_mean(d_z, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
    submean_kernel<<<grid3d,block3d>>>(d_z, mz, g.solid, nx, ny, nz, pitch);
    cudaMemcpy(d_p, d_z, N*sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r,z)
    dot_kernel_1d<<<nb1d,256>>>(d_r, d_z, g.solid, N, d_dot_buf);
    cudaDeviceSynchronize();
    double rsold = host_sum(d_dot_buf, nb1d);
    if (rsold < 1e-30) { cudaMemset(p, 0, N*sizeof(double)); return; }

    for (int k = 0; k < max_iter; k++) {
        // ── FUSED: matvec(tiled) + dot(p,Ap) ──
        matvec_tiled_dot_kernel<<<grid3d,block3d>>>(
            d_p, d_Ap, g.solid, nx, ny, nz, pitch,
            g.idx2, g.idy2, g.idz2, g.diag,
            d_dot_buf);
        cudaDeviceSynchronize();
        double pAp = host_sum(d_dot_buf, n3d);
        if (pAp < 1e-15) break;

        double alpha = rsold / pAp;

        // axpy: x += alpha*p (no dot needed)
        axpy_kernel<<<grid3d,block3d>>>(p, d_p, alpha, g.solid, nx, ny, nz, pitch);

        // ── FUSED: axpy(r, -alpha*Ap) + dot(r,r) ──
        axpy_dot_kernel<<<grid3d,block3d>>>(
            d_r, d_Ap, -alpha, g.solid, nx, ny, nz, pitch,
            d_dot_buf);
        cudaDeviceSynchronize();
        double rsnew = host_sum(d_dot_buf, n3d);

        if (std::sqrt(rsnew) < tol) break;

        precond_->apply_optimized(g, d_r, d_z);

        double mz2 = gpu_mean(d_z, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
        submean_kernel<<<grid3d,block3d>>>(d_z, mz2, g.solid, nx, ny, nz, pitch);
        cudaDeviceSynchronize();

        dot_kernel_1d<<<nb1d,256>>>(d_r, d_z, g.solid, N, d_dot_buf);
        cudaDeviceSynchronize();
        double rz = host_sum(d_dot_buf, nb1d);

        double beta = rz / rsold;
        rsold = rz;

        cudaMemcpy(d_Ap, d_p, N*sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_z, N*sizeof(double), cudaMemcpyDeviceToDevice);
        axpy_kernel<<<grid3d,block3d>>>(d_p, d_Ap, beta, g.solid, nx, ny, nz, pitch);
        cudaDeviceSynchronize();
    }
}

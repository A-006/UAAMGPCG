/**
 * @file cuda_pcg.cu
 * @brief CUDA PCG solver with UAAMG preconditioner.
 *
 * Solves (-nabla^2) p = rhs on GPU.
 *
 * Dot products: each block computes a partial sum (via shared-memory reduction),
 * then HOST sums partials sequentially.
 *
 * Value-modifying kernels (submean, negate) use 2D indexing: interior cells only.
 */
#include "solver/cuda/cuda_pcg.h"
#include <cstdio>
#include <cmath>

__device__ inline int idx2d(int i, int j, int s) { return i + j * s; }

// ── 1D dot-product (reads all non-solid cells; safe: ghost=0 adds nothing) ──
__global__ void dot_partial_kernel(
    const double *a, const double *b, const bool *solid,
    int N, double *d_partial)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x) {
        if (!solid[k]) sum += a[k] * b[k];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_partial[blockIdx.x] = sdata[0];
}

// ── 2D interior-only kernels (match CPU: operate on 1..nx × 1..ny) ──

__global__ void sum_interior_kernel(const double *v, const bool *solid,
    int nx, int ny, int stride, double *part)
{
    __shared__ double s[256];
    int tid = threadIdx.x;
    double sum = 0;
    for (int k = blockIdx.x * blockDim.x + tid; k < nx * ny; k += blockDim.x * gridDim.x) {
        int i = (k % nx) + 1, j = (k / nx) + 1;
        int id = idx2d(i, j, stride);
        if (!solid[id]) sum += v[id];
    }
    s[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void count_interior_kernel(const bool *solid,
    int nx, int ny, int stride, int *part)
{
    __shared__ int s[256];
    int tid = threadIdx.x; int sum = 0;
    for (int k = blockIdx.x * blockDim.x + tid; k < nx * ny; k += blockDim.x * gridDim.x) {
        int i = (k % nx) + 1, j = (k / nx) + 1;
        if (!solid[idx2d(i, j, stride)]) sum++;
    }
    s[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void subtract_mean_kernel(double *v, double mean, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = idx2d(i, j, stride);
    if (!solid[id]) v[id] -= mean;
}

__global__ void negate_kernel(double *v, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = idx2d(i, j, stride);
    if (!solid[id]) v[id] = -v[id];
}

// ── Matvec: Ap = (-nabla^2) p ──
__global__ void matvec_kernel(
    const double *p, double *Ap, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = idx2d(i, j, stride);
    if (solid[id]) { Ap[id] = 0.0; return; }
    double pC = p[id];
    double pL = (i>1 && !solid[idx2d(i-1,j,stride)]) ? p[idx2d(i-1,j,stride)] : pC;
    double pR = (i<nx && !solid[idx2d(i+1,j,stride)]) ? p[idx2d(i+1,j,stride)] : pC;
    double pB = (j>1 && !solid[idx2d(i,j-1,stride)]) ? p[idx2d(i,j-1,stride)] : pC;
    double pT = (j<ny && !solid[idx2d(i,j+1,stride)]) ? p[idx2d(i,j+1,stride)] : pC;
    Ap[id] = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
}

// ── AXPY: y += a * x (interior only) ──
__global__ void axpy_kernel(
    double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = idx2d(i, j, stride);
    if (!solid[id]) y[id] += a * x[id];
}

// ── Host reduction (sequential = deterministic) ──
static double host_reduce(const double *d_partial, int nblocks) {
    std::vector<double> h(nblocks);
    cudaMemcpy(h.data(), d_partial, nblocks * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (double v : h) total += v;
    return total;
}

static int host_reduce_int(const int *d_partial, int nblocks) {
    std::vector<int> h(nblocks);
    cudaMemcpy(h.data(), d_partial, nblocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int v : h) total += v;
    return total;
}

// ── Compute mean of v over INTERIOR fluid cells ──
static double compute_mean(const double *d_v, const bool *d_solid,
    int nx, int ny, int stride, double *d_partial, int *d_count, int nblocks)
{
    sum_interior_kernel<<<nblocks, 256>>>(d_v, d_solid, nx, ny, stride, d_partial);
    cudaDeviceSynchronize();
    double s = host_reduce(d_partial, nblocks);
    count_interior_kernel<<<nblocks, 256>>>(d_solid, nx, ny, stride, d_count);
    cudaDeviceSynchronize();
    int c = host_reduce_int(d_count, nblocks);
    return c > 0 ? s / c : 0.0;
}

// ── CudaPCG ──

void CudaPCG::ensure_buffers(int N) {
    if (N_ >= N) return;
    free_buffers();
    cudaMalloc(&d_r,  N * sizeof(double));
    cudaMalloc(&d_z,  N * sizeof(double));
    cudaMalloc(&d_p,  N * sizeof(double));
    cudaMalloc(&d_Ap, N * sizeof(double));
    int max_blocks = (N + 255) / 256 + 1;
    cudaMalloc(&d_dot_buf, max_blocks * sizeof(double));
    cudaMalloc(&d_count_buf, max_blocks * sizeof(int));
    dot_buf_size_ = max_blocks;
    N_ = N;
}

void CudaPCG::free_buffers() {
    if (d_r)  cudaFree(d_r);
    if (d_z)  cudaFree(d_z);
    if (d_p)  cudaFree(d_p);
    if (d_Ap) cudaFree(d_Ap);
    if (d_dot_buf)   cudaFree(d_dot_buf);
    if (d_count_buf) cudaFree(d_count_buf);
    d_r = d_z = d_p = d_Ap = d_dot_buf = nullptr;
    d_count_buf = nullptr;
    dot_buf_size_ = 0; N_ = 0;
}

void CudaPCG::solve(CudaGrid& g, double* p, double* rhs, int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, stride = g.pitch;
    int N = (nx+2)*(ny+2);
    ensure_buffers(N);
    dim3 block2d(16, 16);
    dim3 grid2d((nx + 15)/16, (ny + 15)/16);
    int nblocks1d = dot_buf_size_;

    precond_->build(g);

    // r = rhs, p = 0
    cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N * sizeof(double));

    // Zero-mean + negate RHS (match CPU PCG: rhs = -(rhs - mean))
    double mr = compute_mean(d_r, g.solid, nx, ny, stride, d_dot_buf, d_count_buf, nblocks1d);
    subtract_mean_kernel<<<grid2d, block2d>>>(d_r, mr, g.solid, nx, ny, stride);
    negate_kernel<<<grid2d, block2d>>>(d_r, g.solid, nx, ny, stride);

    // z = M^{-1} * r
    precond_->apply(g, d_r, d_z);

    // Subtract mean from z, copy to p
    double mz = compute_mean(d_z, g.solid, nx, ny, stride, d_dot_buf, d_count_buf, nblocks1d);
    subtract_mean_kernel<<<grid2d, block2d>>>(d_z, mz, g.solid, nx, ny, stride);
    cudaMemcpy(d_p, d_z, N * sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r, z)
    dot_partial_kernel<<<nblocks1d, 256>>>(d_r, d_z, g.solid, N, d_dot_buf);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    double rsold = host_reduce(d_dot_buf, nblocks1d);
    if (rsold < 1e-30) { cudaMemset(p, 0, N * sizeof(double)); return; }

    for (int k = 0; k < max_iter; k++) {
        matvec_kernel<<<grid2d, block2d>>>(d_p, d_Ap, g.solid, nx, ny, stride, g.idx2, g.idy2, g.diag);

        dot_partial_kernel<<<nblocks1d, 256>>>(d_p, d_Ap, g.solid, N, d_dot_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        double pAp = host_reduce(d_dot_buf, nblocks1d);
        if (pAp < 1e-15) break;

        double alpha = rsold / pAp;
        axpy_kernel<<<grid2d, block2d>>>(p, d_p, alpha, g.solid, nx, ny, stride);
        axpy_kernel<<<grid2d, block2d>>>(d_r, d_Ap, -alpha, g.solid, nx, ny, stride);
        CUDA_CHECK(cudaDeviceSynchronize());

        dot_partial_kernel<<<nblocks1d, 256>>>(d_r, d_r, g.solid, N, d_dot_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        double rsnew = host_reduce(d_dot_buf, nblocks1d);

        if (std::sqrt(rsnew) < tol) break;

        precond_->apply(g, d_r, d_z);

        double mz2 = compute_mean(d_z, g.solid, nx, ny, stride, d_dot_buf, d_count_buf, nblocks1d);
        subtract_mean_kernel<<<grid2d, block2d>>>(d_z, mz2, g.solid, nx, ny, stride);
        CUDA_CHECK(cudaDeviceSynchronize());

        dot_partial_kernel<<<nblocks1d, 256>>>(d_r, d_z, g.solid, N, d_dot_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        double rz = host_reduce(d_dot_buf, nblocks1d);

        double beta = rz / rsold;
        rsold = rz;

        // p = z + beta * p
        cudaMemcpy(d_Ap, d_p, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_z, N * sizeof(double), cudaMemcpyDeviceToDevice);
        axpy_kernel<<<grid2d, block2d>>>(d_p, d_Ap, beta, g.solid, nx, ny, stride);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @file cuda_pcg_3d.cu
 * @brief CUDA 3D PCG solver with UAAMG preconditioner.
 *
 * Solves (-nabla^2) p = rhs on GPU with 7-point stencil.
 * 3D extension of cuda_pcg.cu.
 */
#include "solver/cuda/cuda_pcg_3d.h"
#include <cstdio>
#include <cmath>

__device__ inline int idx3d(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

// ── 1D dot-product (solid cells skipped) ──
__global__ void dot_partial_kernel_3d(
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

// ── 3D interior sum (1D thread mapping over nx*ny*nz) ──
__global__ void sum_interior_kernel_3d(const double *v, const bool *solid,
    int nx, int ny, int nz, int pitch, double *part)
{
    __shared__ double s[256];
    int tid = threadIdx.x;
    double sum = 0;
    int total_threads = blockDim.x * gridDim.x;
    int interior = nx * ny * nz;
    for (int lin = blockIdx.x * blockDim.x + tid; lin < interior; lin += total_threads) {
        int i = (lin % nx) + 1;
        int j = ((lin / nx) % ny) + 1;
        int k = (lin / (nx * ny)) + 1;
        int id = idx3d(i, j, k, pitch, ny);
        if (!solid[id]) sum += v[id];
    }
    s[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

// ── 3D interior count (1D thread mapping) ──
__global__ void count_interior_kernel_3d(const bool *solid,
    int nx, int ny, int nz, int pitch, int *part)
{
    __shared__ int s[256];
    int tid = threadIdx.x; int sum = 0;
    int total_threads = blockDim.x * gridDim.x;
    int interior = nx * ny * nz;
    for (int lin = blockIdx.x * blockDim.x + tid; lin < interior; lin += total_threads) {
        int i = (lin % nx) + 1;
        int j = ((lin / nx) % ny) + 1;
        int k = (lin / (nx * ny)) + 1;
        if (!solid[idx3d(i, j, k, pitch, ny)]) sum++;
    }
    s[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

// ── 3D subtract mean (interior only) ──
__global__ void subtract_mean_kernel_3d(double *v, double mean, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id]) v[id] -= mean;
}

// ── 3D negate (interior only) ──
__global__ void negate_kernel_3d(double *v, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id]) v[id] = -v[id];
}

// ── 3D Matvec: Ap = (-nabla^2) p, 7-point stencil ──
__global__ void matvec_kernel_3d(
    const double *p, double *Ap, const bool *solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (solid[id]) { Ap[id] = 0.0; return; }

    double pC = p[id];
    double pL = (i>1   && !solid[idx3d(i-1, j,   k,   pitch, ny)]) ? p[idx3d(i-1, j,   k,   pitch, ny)] : pC;
    double pR = (i<nx  && !solid[idx3d(i+1, j,   k,   pitch, ny)]) ? p[idx3d(i+1, j,   k,   pitch, ny)] : pC;
    double pB = (j>1   && !solid[idx3d(i,   j-1, k,   pitch, ny)]) ? p[idx3d(i,   j-1, k,   pitch, ny)] : pC;
    double pT = (j<ny  && !solid[idx3d(i,   j+1, k,   pitch, ny)]) ? p[idx3d(i,   j+1, k,   pitch, ny)] : pC;
    double pF = (k>1   && !solid[idx3d(i,   j,   k-1, pitch, ny)]) ? p[idx3d(i,   j,   k-1, pitch, ny)] : pC;
    double pK = (k<nz  && !solid[idx3d(i,   j,   k+1, pitch, ny)]) ? p[idx3d(i,   j,   k+1, pitch, ny)] : pC;

    Ap[id] = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2 - (pF+pK)*idz2;
}

// ── 3D AXPY: y += a * x (interior only) ──
__global__ void axpy_kernel_3d(
    double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id]) y[id] += a * x[id];
}

// ── Host reduction (deterministic sequential) ──
static double host_reduce_3d(const double *d_partial, int nblocks) {
    std::vector<double> h(nblocks);
    cudaMemcpy(h.data(), d_partial, nblocks * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (double v : h) total += v;
    return total;
}

static int host_reduce_int_3d(const int *d_partial, int nblocks) {
    std::vector<int> h(nblocks);
    cudaMemcpy(h.data(), d_partial, nblocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int v : h) total += v;
    return total;
}

// ── Compute mean of v over INTERIOR fluid cells ──
static double compute_mean_3d(const double *d_v, const bool *d_solid,
    int nx, int ny, int nz, int pitch,
    double *d_partial, int *d_count, int nblocks)
{
    sum_interior_kernel_3d<<<nblocks, 256>>>(d_v, d_solid, nx, ny, nz, pitch, d_partial);
    cudaDeviceSynchronize();
    double s = host_reduce_3d(d_partial, nblocks);
    count_interior_kernel_3d<<<nblocks, 256>>>(d_solid, nx, ny, nz, pitch, d_count);
    cudaDeviceSynchronize();
    int c = host_reduce_int_3d(d_count, nblocks);
    return c > 0 ? s / c : 0.0;
}

// ── CudaPCG3D ──

void CudaPCG3D::ensure_buffers(int N) {
    if (N_ >= N) return;
    free_buffers();
    cudaMalloc(&d_r,  N * sizeof(double));
    cudaMalloc(&d_z,  N * sizeof(double));
    cudaMalloc(&d_p,  N * sizeof(double));
    cudaMalloc(&d_Ap, N * sizeof(double));
    int max_blocks = (N + 255) / 256 + 1;
    cudaMalloc(&d_dot_buf,   max_blocks * sizeof(double));
    cudaMalloc(&d_count_buf, max_blocks * sizeof(int));
	cudaMalloc(&d_scalar, sizeof(double));
    dot_buf_size_ = max_blocks;
    N_ = N;
}

void CudaPCG3D::free_buffers() {
    if (d_r)  cudaFree(d_r);
    if (d_z)  cudaFree(d_z);
    if (d_p)  cudaFree(d_p);
    if (d_Ap) cudaFree(d_Ap);
    if (d_dot_buf)   cudaFree(d_dot_buf);
    if (d_scalar)  cudaFree(d_scalar);
    if (d_count_buf) cudaFree(d_count_buf);
    d_r = d_z = d_p = d_Ap = d_dot_buf = nullptr;
    d_count_buf = nullptr; d_scalar = nullptr;
    dot_buf_size_ = 0; N_ = 0;
}

void CudaPCG3D::solve(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, nz = g.nz, pitch = g.pitch;
    int N = (nx+2)*(ny+2)*(nz+2);
    ensure_buffers(N);
    dim3 block3d(8, 8, 8);
    dim3 grid3d((nx + 7)/8, (ny + 7)/8, (nz + 7)/8);
    int nblocks1d = dot_buf_size_;

    precond_->build(g);

    // r = rhs, p = 0
    cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N * sizeof(double));

    // Zero-mean + negate RHS (match CPU PCG: rhs = -(rhs - mean))
    double mr = compute_mean_3d(d_r, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nblocks1d);
    subtract_mean_kernel_3d<<<grid3d, block3d>>>(d_r, mr, g.solid, nx, ny, nz, pitch);
    negate_kernel_3d<<<grid3d, block3d>>>(d_r, g.solid, nx, ny, nz, pitch);

    // z = M^{-1} * r
    precond_->apply(g, d_r, d_z);

    // Subtract mean from z, copy to p
    double mz = compute_mean_3d(d_z, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nblocks1d);
    subtract_mean_kernel_3d<<<grid3d, block3d>>>(d_z, mz, g.solid, nx, ny, nz, pitch);
    cudaMemcpy(d_p, d_z, N * sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r, z)
    dot_partial_kernel_3d<<<nblocks1d, 256>>>(d_r, d_z, g.solid, N, d_dot_buf);
    CUDA_CHECK_3D(cudaGetLastError());
    cudaDeviceSynchronize();
    double rsold = host_reduce_3d(d_dot_buf, nblocks1d);
    if (rsold < 1e-30) { cudaMemset(p, 0, N * sizeof(double)); return; }

    for (int k = 0; k < max_iter; k++) {
        matvec_kernel_3d<<<grid3d, block3d>>>(
            d_p, d_Ap, g.solid, nx, ny, nz, pitch, g.idx2, g.idy2, g.idz2, g.diag);

        dot_partial_kernel_3d<<<nblocks1d, 256>>>(d_p, d_Ap, g.solid, N, d_dot_buf);
        CUDA_CHECK_3D(cudaDeviceSynchronize());
        double pAp = host_reduce_3d(d_dot_buf, nblocks1d);
        if (pAp < 1e-15) break;

        double alpha = rsold / pAp;

        axpy_kernel_3d<<<grid3d, block3d>>>(p, d_p, alpha, g.solid, nx, ny, nz, pitch);
        axpy_kernel_3d<<<grid3d, block3d>>>(d_r, d_Ap, -alpha, g.solid, nx, ny, nz, pitch);
        CUDA_CHECK_3D(cudaDeviceSynchronize());

        dot_partial_kernel_3d<<<nblocks1d, 256>>>(d_r, d_r, g.solid, N, d_dot_buf);
        CUDA_CHECK_3D(cudaDeviceSynchronize());
        double rsnew = host_reduce_3d(d_dot_buf, nblocks1d);

        if (std::sqrt(rsnew) < tol) break;

        precond_->apply(g, d_r, d_z);

        double mz2 = compute_mean_3d(d_z, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nblocks1d);
        subtract_mean_kernel_3d<<<grid3d, block3d>>>(d_z, mz2, g.solid, nx, ny, nz, pitch);
        CUDA_CHECK_3D(cudaDeviceSynchronize());

        dot_partial_kernel_3d<<<nblocks1d, 256>>>(d_r, d_z, g.solid, N, d_dot_buf);
        CUDA_CHECK_3D(cudaDeviceSynchronize());
        double rz = host_reduce_3d(d_dot_buf, nblocks1d);

        double beta = rz / rsold;
        rsold = rz;

        // p = z + beta * p
        cudaMemcpy(d_Ap, d_p, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_z, N * sizeof(double), cudaMemcpyDeviceToDevice);
        axpy_kernel_3d<<<grid3d, block3d>>>(d_p, d_Ap, beta, g.solid, nx, ny, nz, pitch);
        CUDA_CHECK_3D(cudaDeviceSynchronize());
    }
    CUDA_CHECK_3D(cudaDeviceSynchronize());
}

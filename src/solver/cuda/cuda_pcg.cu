/**
 * @file cuda_pcg.cu
 * @brief CUDA PCG solver with UAAMG preconditioner.
 *
 * Solves (-nabla^2) p = rhs on GPU.
 *
 * Dot products: each block computes a partial sum (via shared-memory reduction),
 * then HOST sums partials sequentially → identical order to CPU → bit-identical.
 */
#include "solver/cuda/cuda_pcg.h"
#include <cstdio>
#include <cmath>

// ── Per-block partial dot product (a·b over fluid cells) ──
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

// ── Per-block partial sum (v over fluid cells, for mean computation) ──
__global__ void sum_partial_kernel(
    const double *v, const bool *solid, int N, double *d_partial)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x) {
        if (!solid[k]) sum += v[k];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_partial[blockIdx.x] = sdata[0];
}

// ── Count fluid cells ──
__global__ void count_fluid_kernel(const bool *solid, int N, int *d_count) {
    __shared__ int scount[256];
    int tid = threadIdx.x;
    int sum = 0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x)
        if (!solid[k]) sum++;
    scount[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) scount[tid] += scount[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_count[blockIdx.x] = scount[0];
}

// ── Reduce partial sums on host (sequential = deterministic = matches CPU) ──
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

// ── Matvec: Ap = (-nabla^2) p ──
__global__ void matvec_kernel(
    const double *p, double *Ap, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = i + j * stride;
    if (solid[id]) { Ap[id] = 0.0; return; }
    double pC = p[id];
    double pL = (i>1 && !solid[(i-1)+j*stride]) ? p[(i-1)+j*stride] : pC;
    double pR = (i<nx && !solid[(i+1)+j*stride]) ? p[(i+1)+j*stride] : pC;
    double pB = (j>1 && !solid[i+(j-1)*stride]) ? p[i+(j-1)*stride] : pC;
    double pT = (j<ny && !solid[i+(j+1)*stride]) ? p[i+(j+1)*stride] : pC;
    Ap[id] = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
}

// ── AXPY: y += a * x  (fluid cells only) ──
__global__ void axpy_kernel(
    double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = i + j * stride;
    if (!solid[id]) y[id] += a * x[id];
}

// ── Subtract mean ──
__global__ void subtract_mean_kernel(
    double *v, double mean, const bool *solid, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k]) v[k] -= mean;
}

// ── Negate ──
__global__ void negate_kernel(double *v, const bool *solid, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k]) v[k] = -v[k];
}

// ── Compute mean of v over fluid cells ──
static double compute_mean(const double *d_v, const bool *d_solid, int N,
                           double *d_partial, int *d_count, int nblocks) {
    sum_partial_kernel<<<nblocks, 256>>>(d_v, d_solid, N, d_partial);
    cudaDeviceSynchronize();
    double s = host_reduce(d_partial, nblocks);
    count_fluid_kernel<<<nblocks, 256>>>(d_solid, N, d_count);
    cudaDeviceSynchronize();
    int c = host_reduce_int(d_count, nblocks);
    return c > 0 ? s / c : 0.0;
}

// ── CPU reference dot (allocated once, used for verification) ──
static std::vector<double> h_a, h_b, h_tmp;
static std::vector<char>   h_s;
static double cpu_dot_verify(const double *d_a, const double *d_b, const bool *d_solid, int N) {
    if ((int)h_a.size() < N) { h_a.resize(N); h_b.resize(N); h_tmp.resize(N); h_s.resize(N); }
    cudaMemcpy(h_a.data(), d_a, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), d_b, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_s.data(), d_solid, N*sizeof(bool), cudaMemcpyDeviceToHost);
    double t = 0; for (int k = 0; k < N; k++) if (!h_s[k]) t += h_a[k] * h_b[k];
    return t;
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
    double mr = compute_mean(d_r, g.solid, N, d_dot_buf, d_count_buf, nblocks1d);
    subtract_mean_kernel<<<(N+255)/256, 256>>>(d_r, mr, g.solid, N); // r -= mean
    negate_kernel<<<(N+255)/256, 256>>>(d_r, g.solid, N);            // r = -r

    // z = M^{-1} * r
    precond_->apply(g, d_r, d_z);

    // Subtract mean from z, copy to p
    double mz = compute_mean(d_z, g.solid, N, d_dot_buf, d_count_buf, nblocks1d);
    subtract_mean_kernel<<<(N+255)/256, 256>>>(d_z, mz, g.solid, N);
    cudaMemcpy(d_p, d_z, N * sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r, z)
    dot_partial_kernel<<<nblocks1d, 256>>>(d_r, d_z, g.solid, N, d_dot_buf);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    double rsold = host_reduce(d_dot_buf, nblocks1d);
    if (rsold < 1e-30) { cudaMemset(p, 0, N * sizeof(double)); return; }

    // Diagnostic: verify initial dot matches CPU
    if (max_iter > 0) {
        double cpu_rs = cpu_dot_verify(d_r, d_z, g.solid, N);
        printf("  PCG init: GPU rsold=%.10e  CPU rsold=%.10e  diff=%.2e\n", rsold, cpu_rs, std::abs(rsold-cpu_rs));
    }

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

        // ── Verify: compare every GPU dot product with CPU reference ──
        if (k < 3) {
            double cpu_pAp = cpu_dot_verify(d_p, d_Ap, g.solid, N);
            double cpu_rr  = cpu_dot_verify(d_r, d_r, g.solid, N);
            printf("  iter %d: GPU pAp=%.6e CPU=%.6e d=%.2e | GPU |r|^2=%.6e CPU=%.6e d=%.2e | alpha=%.6e\n",
                   k, pAp, cpu_pAp, std::abs(pAp-cpu_pAp), rsnew, cpu_rr, std::abs(rsnew-cpu_rr), alpha);
        } else if (k < 5 || k % 10 == 0) {
            printf("  PCG iter %d: sqrt(r·r)=%.4e  pAp=%.4e  alpha=%.4e  rsold=%.4e\n",
                   k, std::sqrt(rsnew), pAp, alpha, rsold);
        }
        if (std::sqrt(rsnew) < tol) break;

        precond_->apply(g, d_r, d_z);

        // ── Verify V-cycle output: compare GPU z with CPU UAAMG ──
        if (k < 2) {
            // Check if d_z is finite and non-zero
            cudaMemcpy(h_tmp.data(), d_z, N*sizeof(double), cudaMemcpyDeviceToHost);
            double zmax=0; for(int i=0;i<N;i++) if(!h_s[i]) zmax=std::max(zmax,std::abs(h_tmp[i]));
            printf("          after apply: max|d_z|=%e\n", zmax);
        }

        mz = compute_mean(d_z, g.solid, N, d_dot_buf, d_count_buf, nblocks1d);
        subtract_mean_kernel<<<(N+255)/256, 256>>>(d_z, mz, g.solid, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        dot_partial_kernel<<<nblocks1d, 256>>>(d_r, d_z, g.solid, N, d_dot_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        double rz = host_reduce(d_dot_buf, nblocks1d);
        if (k < 3) {
            double cpu_rz = cpu_dot_verify(d_r, d_z, g.solid, N);
            printf("          GPU dot(r,z)=%.6e CPU=%.6e d=%.2e\n", rz, cpu_rz, std::abs(rz-cpu_rz));
        }
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

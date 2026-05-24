/**
 * @file cuda_cg.cu
 * @brief CUDA Conjugate Gradient (no preconditioner, M=I).
 *
 * Simplest possible GPU solver — just PCG with identity.
 * Used to verify GPU PCG loop correctness before adding UAAMG.
 */
#include "solver/cuda/cuda_cg.h"
#include <vector>

// ── GPU kernels ──

__device__ inline int didx(int i, int j, int s) { return i + j * s; }

__global__ void cg_matvec(const double *p, double *Ap, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = didx(i,j,stride);
    if (solid[id]) { Ap[id] = 0.0; return; }
    double pC = p[id];
    double pL = (i>1 && !solid[didx(i-1,j,stride)]) ? p[didx(i-1,j,stride)] : pC;
    double pR = (i<nx && !solid[didx(i+1,j,stride)]) ? p[didx(i+1,j,stride)] : pC;
    double pB = (j>1 && !solid[didx(i,j-1,stride)]) ? p[didx(i,j-1,stride)] : pC;
    double pT = (j<ny && !solid[didx(i,j+1,stride)]) ? p[didx(i,j+1,stride)] : pC;
    Ap[id] = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
}

__global__ void cg_axpy(double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = didx(i,j,stride);
    if (!solid[id]) y[id] += a * x[id];
}

// ── 2D interior-only kernels (match CPU: operate only on 1..nx × 1..ny) ──

__global__ void cg_submean_2d(double *v, double m, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = didx(i,j,stride);
    if (!solid[id]) v[id] -= m;
}

__global__ void cg_negate_2d(double *v, const bool *solid,
    int nx, int ny, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = didx(i,j,stride);
    if (!solid[id]) v[id] = -v[id];
}

__global__ void cg_sum_2d(const double *v, const bool *solid,
    int nx, int ny, int stride, double *part)
{
    __shared__ double s[256];
    int tid = threadIdx.x;
    double sum = 0;
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    // Map 1D thread index to 2D interior cells
    for (int k = idx; k < nx * ny; k += total_threads) {
        int i = (k % nx) + 1;
        int j = (k / nx) + 1;
        int id = didx(i,j,stride);
        if (!solid[id]) sum += v[id];
    }
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void cg_count_2d(const bool *solid, int nx, int ny, int stride, int *part) {
    __shared__ int s[256]; int tid = threadIdx.x; int sum = 0;
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    for (int k = idx; k < nx * ny; k += total_threads) {
        int i = (k % nx) + 1;
        int j = (k / nx) + 1;
        if (!solid[didx(i,j,stride)]) sum++;
    }
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    if (tid == 0) part[blockIdx.x] = s[0];
}

// ── 1D dot-product (reads all non-solid cells; safe as long as ghost=0) ──
__global__ void cg_dot_partial(const double *a, const double *b, const bool *solid,
    int N, double *part)
{
    __shared__ double s[256];
    int tid = threadIdx.x;
    double sum = 0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x)
        if (!solid[k]) sum += a[k] * b[k];
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

// ── Host helpers ──

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

// Compute mean over INTERIOR cells only (1..nx × 1..ny)
static double gpu_mean(const double *v, const bool *solid,
    int nx, int ny, int stride, double *part, int *cnt, int nb)
{
    cg_sum_2d<<<nb,256>>>(v, solid, nx, ny, stride, part); cudaDeviceSynchronize();
    double s = host_sum(part, nb);
    cg_count_2d<<<nb,256>>>(solid, nx, ny, stride, cnt); cudaDeviceSynchronize();
    int c = host_sum_int(cnt, nb);
    return c > 0 ? s / c : 0;
}

// ── CudaCG ──

void CudaCG::ensure_buffers(int N) {
    if (N_ >= N) return;
    free_buffers();
    cudaMalloc(&d_r,   N * sizeof(double));
    cudaMalloc(&d_p,   N * sizeof(double));
    cudaMalloc(&d_Ap,  N * sizeof(double));
    int nb = (N + 255) / 256 + 1;
    cudaMalloc(&d_dot_buf,   nb * sizeof(double));
    cudaMalloc(&d_count_buf, nb * sizeof(int));
    dot_buf_size_ = nb; N_ = N;
}

void CudaCG::free_buffers() {
    if (d_r) cudaFree(d_r);
    if (d_p) cudaFree(d_p);
    if (d_Ap) cudaFree(d_Ap);
    if (d_dot_buf) cudaFree(d_dot_buf);
    if (d_count_buf) cudaFree(d_count_buf);
    d_r = d_p = d_Ap = d_dot_buf = nullptr;
    d_count_buf = nullptr; dot_buf_size_ = 0; N_ = 0;
}

void CudaCG::solve(CudaGrid& g, double* p, double* rhs, int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, stride = g.pitch, N = (nx+2)*(ny+2);
    ensure_buffers(N);
    dim3 block2d(16,16), grid2d((nx+15)/16,(ny+15)/16);
    int nb1d = (N + 255) / 256 + 1;

    // r = rhs, p = 0
    cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N * sizeof(double));

    // Zero-mean + negate: r = -(rhs - mean(rhs)) — only interior cells
    double mr = gpu_mean(d_r, g.solid, nx, ny, stride, d_dot_buf, d_count_buf, nb1d);
    cg_submean_2d<<<grid2d,block2d>>>(d_r, mr, g.solid, nx, ny, stride);
    cg_negate_2d<<<grid2d,block2d>>>(d_r, g.solid, nx, ny, stride);
    cudaDeviceSynchronize();

    // CG: p = r (identity preconditioner)
    cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r, r)
    cg_dot_partial<<<nb1d,256>>>(d_r, d_r, g.solid, N, d_dot_buf);
    cudaDeviceSynchronize();
    double rsold = host_sum(d_dot_buf, nb1d);

    for (int k = 0; k < max_iter; k++) {
        cg_matvec<<<grid2d,block2d>>>(d_p, d_Ap, g.solid, nx, ny, stride, g.idx2, g.idy2, g.diag);
        cudaDeviceSynchronize();

        cg_dot_partial<<<nb1d,256>>>(d_p, d_Ap, g.solid, N, d_dot_buf);
        cudaDeviceSynchronize();
        double pAp = host_sum(d_dot_buf, nb1d);
        if (pAp < 1e-15) break;

        double alpha = rsold / pAp;

        cg_axpy<<<grid2d,block2d>>>(p, d_p, alpha, g.solid, nx, ny, stride);
        cg_axpy<<<grid2d,block2d>>>(d_r, d_Ap, -alpha, g.solid, nx, ny, stride);
        cudaDeviceSynchronize();

        cg_dot_partial<<<nb1d,256>>>(d_r, d_r, g.solid, N, d_dot_buf);
        cudaDeviceSynchronize();
        double rsnew = host_sum(d_dot_buf, nb1d);

        if (std::sqrt(rsnew) < tol) break;

        double beta = rsnew / rsold;
        rsold = rsnew;

        // p_new = r + beta * p_old
        // d_Ap already contains matvec result; overwrite with p_old for axpy
        cudaMemcpy(d_Ap, d_p, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cg_axpy<<<grid2d,block2d>>>(d_p, d_Ap, beta, g.solid, nx, ny, stride);
        cudaDeviceSynchronize();
    }
}

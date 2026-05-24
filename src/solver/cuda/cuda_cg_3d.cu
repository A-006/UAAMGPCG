/**
 * @file cuda_cg_3d.cu
 * @brief CUDA 3D Conjugate Gradient (no preconditioner, M=I).
 *
 * 3D extension of cuda_cg.cu with 7-point stencil and 3D thread blocks.
 */
#include "solver/cuda/cuda_cg_3d.h"
#include <vector>

// ── 3D index: pitch = nx+2, stride_yz = pitch * (ny+2) ──
__device__ inline int idx3d(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

// ── 3D matvec: 7-point Laplacian stencil ──
__global__ void cg3d_matvec(const double *p, double *Ap, const bool *solid,
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

// ── 3D AXPY: y += a * x (interior cells only) ──
__global__ void cg3d_axpy(double *y, const double *x, double a, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id]) y[id] += a * x[id];
}

// ── Subtract mean from interior cells ──
__global__ void cg3d_submean_3d(double *v, double m, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id]) v[id] -= m;
}

// ── Negate interior cells ──
__global__ void cg3d_negate_3d(double *v, const bool *solid,
    int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz) return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id]) v[id] = -v[id];
}

// ── Sum over interior cells (1D thread mapping over nx*ny*nz) ──
__global__ void cg3d_sum_3d(const double *v, const bool *solid,
    int nx, int ny, int nz, int pitch, double *part)
{
    __shared__ double s[256];
    int tid = threadIdx.x;
    double sum = 0;
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int interior = nx * ny * nz;
    for (int lin = idx; lin < interior; lin += total_threads) {
        int i = (lin % nx) + 1;
        int j = ((lin / nx) % ny) + 1;
        int k = (lin / (nx * ny)) + 1;
        int id = idx3d(i, j, k, pitch, ny);
        if (!solid[id]) sum += v[id];
    }
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

// ── Count interior fluid cells (1D thread mapping) ──
__global__ void cg3d_count_3d(const bool *solid,
    int nx, int ny, int nz, int pitch, int *part)
{
    __shared__ int s[256];
    int tid = threadIdx.x;
    int sum = 0;
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int interior = nx * ny * nz;
    for (int lin = idx; lin < interior; lin += total_threads) {
        int i = (lin % nx) + 1;
        int j = ((lin / nx) % ny) + 1;
        int k = (lin / (nx * ny)) + 1;
        if (!solid[idx3d(i, j, k, pitch, ny)]) sum++;
    }
    s[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

// ── 1D dot-product over all cells (ghost=0, solid skipped) ──
__global__ void cg3d_dot_partial(const double *a, const double *b, const bool *solid,
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

// Compute mean over INTERIOR cells only
static double gpu_mean_3d(const double *v, const bool *solid,
    int nx, int ny, int nz, int pitch, double *part, int *cnt, int nb)
{
    cg3d_sum_3d<<<nb,256>>>(v, solid, nx, ny, nz, pitch, part); cudaDeviceSynchronize();
    double s = host_sum(part, nb);
    cg3d_count_3d<<<nb,256>>>(solid, nx, ny, nz, pitch, cnt); cudaDeviceSynchronize();
    int c = host_sum_int(cnt, nb);
    return c > 0 ? s / c : 0;
}

// ── CudaCG3D ──

void CudaCG3D::ensure_buffers(int N) {
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

void CudaCG3D::free_buffers() {
    if (d_r)   cudaFree(d_r);
    if (d_p)   cudaFree(d_p);
    if (d_Ap)  cudaFree(d_Ap);
    if (d_dot_buf)   cudaFree(d_dot_buf);
    if (d_count_buf) cudaFree(d_count_buf);
    d_r = d_p = d_Ap = d_dot_buf = nullptr;
    d_count_buf = nullptr; dot_buf_size_ = 0; N_ = 0;
}

void CudaCG3D::solve(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, nz = g.nz, pitch = g.pitch;
    int N = (nx+2)*(ny+2)*(nz+2);
    ensure_buffers(N);
    dim3 block3d(8, 8, 8);
    dim3 grid3d((nx+7)/8, (ny+7)/8, (nz+7)/8);
    int nb1d = (N + 255) / 256 + 1;

    // r = rhs, p = 0
    cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(p, 0, N * sizeof(double));

    // Zero-mean + negate: r = -(rhs - mean(rhs)) — interior only
    double mr = gpu_mean_3d(d_r, g.solid, nx, ny, nz, pitch, d_dot_buf, d_count_buf, nb1d);
    cg3d_submean_3d<<<grid3d, block3d>>>(d_r, mr, g.solid, nx, ny, nz, pitch);
    cg3d_negate_3d<<<grid3d, block3d>>>(d_r, g.solid, nx, ny, nz, pitch);
    cudaDeviceSynchronize();

    // CG: p = r (identity preconditioner)
    cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice);

    // rsold = dot(r, r)
    cg3d_dot_partial<<<nb1d,256>>>(d_r, d_r, g.solid, N, d_dot_buf);
    cudaDeviceSynchronize();
    double rsold = host_sum(d_dot_buf, nb1d);

    for (int k = 0; k < max_iter; k++) {
        cg3d_matvec<<<grid3d, block3d>>>(d_p, d_Ap, g.solid, nx, ny, nz, pitch,
            g.idx2, g.idy2, g.idz2, g.diag);
        cudaDeviceSynchronize();

        cg3d_dot_partial<<<nb1d,256>>>(d_p, d_Ap, g.solid, N, d_dot_buf);
        cudaDeviceSynchronize();
        double pAp = host_sum(d_dot_buf, nb1d);
        if (pAp < 1e-15) break;

        double alpha = rsold / pAp;

        cg3d_axpy<<<grid3d, block3d>>>(p, d_p, alpha, g.solid, nx, ny, nz, pitch);
        cg3d_axpy<<<grid3d, block3d>>>(d_r, d_Ap, -alpha, g.solid, nx, ny, nz, pitch);
        cudaDeviceSynchronize();

        cg3d_dot_partial<<<nb1d,256>>>(d_r, d_r, g.solid, N, d_dot_buf);
        cudaDeviceSynchronize();
        double rsnew = host_sum(d_dot_buf, nb1d);

        if (std::sqrt(rsnew) < tol) break;

        double beta = rsnew / rsold;
        rsold = rsnew;

        // p_new = r + beta * p_old
        cudaMemcpy(d_Ap, d_p, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice);
        cg3d_axpy<<<grid3d, block3d>>>(d_p, d_Ap, beta, g.solid, nx, ny, nz, pitch);
        cudaDeviceSynchronize();
    }
}

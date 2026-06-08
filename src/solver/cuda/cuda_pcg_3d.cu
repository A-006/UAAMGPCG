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
__global__ void dot_partial_kernel_3d(const double* a, const double* b, const bool* solid, int N,
                                      double* d_partial) {
    __shared__ double sdata[256];
    int tid    = threadIdx.x;
    double sum = 0.0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x) {
        if (!solid[k])
            sum += a[k] * b[k];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        d_partial[blockIdx.x] = sdata[0];
}

// ── 3D interior sum (1D thread mapping over nx*ny*nz) ──
__global__ void sum_interior_kernel_3d(const double* v, const bool* solid, int nx, int ny, int nz,
                                       int pitch, double* part) {
    __shared__ double s[256];
    int tid           = threadIdx.x;
    double sum        = 0;
    int total_threads = blockDim.x * gridDim.x;
    int interior      = nx * ny * nz;
    for (int lin = blockIdx.x * blockDim.x + tid; lin < interior; lin += total_threads) {
        int i  = (lin % nx) + 1;
        int j  = ((lin / nx) % ny) + 1;
        int k  = (lin / (nx * ny)) + 1;
        int id = idx3d(i, j, k, pitch, ny);
        if (!solid[id])
            sum += v[id];
    }
    s[tid] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0)
        part[blockIdx.x] = s[0];
}

// ── 3D interior count (1D thread mapping) ──
__global__ void count_interior_kernel_3d(const bool* solid, int nx, int ny, int nz, int pitch,
                                         int* part) {
    __shared__ int s[256];
    int tid           = threadIdx.x;
    int sum           = 0;
    int total_threads = blockDim.x * gridDim.x;
    int interior      = nx * ny * nz;
    for (int lin = blockIdx.x * blockDim.x + tid; lin < interior; lin += total_threads) {
        int i = (lin % nx) + 1;
        int j = ((lin / nx) % ny) + 1;
        int k = (lin / (nx * ny)) + 1;
        if (!solid[idx3d(i, j, k, pitch, ny)])
            sum++;
    }
    s[tid] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0)
        part[blockIdx.x] = s[0];
}

// ── 3D subtract mean (interior only) ──
__global__ void subtract_mean_kernel_3d(double* v, double mean, const bool* solid, int nx, int ny,
                                        int nz, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id])
        v[id] -= mean;
}

// ── 3D negate (interior only) ──
__global__ void negate_kernel_3d(double* v, const bool* solid, int nx, int ny, int nz, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id])
        v[id] = -v[id];
}

// ── 3D Matvec: Ap = (-nabla^2) p, 7-point stencil ──
__global__ void matvec_kernel_3d(const double* p, double* Ap, const bool* solid, int nx, int ny,
                                 int nz, int pitch, double idx2, double idy2, double idz2,
                                 double diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int id = idx3d(i, j, k, pitch, ny);
    if (solid[id]) {
        Ap[id] = 0.0;
        return;
    }

    double pC = p[id];
    double pL =
        (i > 1 && !solid[idx3d(i - 1, j, k, pitch, ny)]) ? p[idx3d(i - 1, j, k, pitch, ny)] : pC;
    double pR =
        (i < nx && !solid[idx3d(i + 1, j, k, pitch, ny)]) ? p[idx3d(i + 1, j, k, pitch, ny)] : pC;
    double pB =
        (j > 1 && !solid[idx3d(i, j - 1, k, pitch, ny)]) ? p[idx3d(i, j - 1, k, pitch, ny)] : pC;
    double pT =
        (j < ny && !solid[idx3d(i, j + 1, k, pitch, ny)]) ? p[idx3d(i, j + 1, k, pitch, ny)] : pC;
    double pF =
        (k > 1 && !solid[idx3d(i, j, k - 1, pitch, ny)]) ? p[idx3d(i, j, k - 1, pitch, ny)] : pC;
    double pK =
        (k < nz && !solid[idx3d(i, j, k + 1, pitch, ny)]) ? p[idx3d(i, j, k + 1, pitch, ny)] : pC;

    Ap[id] = diag * pC - (pL + pR) * idx2 - (pB + pT) * idy2 - (pF + pK) * idz2;
}

// ── 3D AXPY: y += a * x (interior only) ──
__global__ void axpy_kernel_3d(double* y, const double* x, double a, const bool* solid, int nx,
                               int ny, int nz, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int id = idx3d(i, j, k, pitch, ny);
    if (!solid[id])
        y[id] += a * x[id];
}

// ── Flat (tile-layout) vector ops: iterate the contiguous num_tiles*512 array with
//    the tile-layout solid mask. Valid because all grid dims are multiples of 8, so
//    the tile array is exactly the interior (no padding cells). Used by the
//    tile-native solve path so r/z/p/Ap never round-trip through scatter/gather. ──
__global__ void negate_flat_kernel(double* v, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        v[k] = -v[k];
}
__global__ void subtract_mean_flat_kernel(double* v, double mean, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        v[k] -= mean;
}
__global__ void axpy_flat_kernel(double* y, const double* x, double a, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        y[k] += a * x[k];
}
// y = x + b*y  (PCG direction update p = z + beta*p)
__global__ void xpby_flat_kernel(double* y, const double* x, double b, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        y[k] = x[k] + b * y[k];
}
__global__ void sum_flat_kernel(const double* v, const bool* solid, long N, double* part) {
    __shared__ double s[256];
    int tid    = threadIdx.x;
    double sum = 0;
    for (long k = (long)blockIdx.x * blockDim.x + tid; k < N; k += (long)blockDim.x * gridDim.x)
        if (!solid[k])
            sum += v[k];
    s[tid] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0)
        part[blockIdx.x] = s[0];
}
__global__ void count_flat_kernel(const bool* solid, long N, int* part) {
    __shared__ int s[256];
    int tid = threadIdx.x;
    int c   = 0;
    for (long k = (long)blockIdx.x * blockDim.x + tid; k < N; k += (long)blockDim.x * gridDim.x)
        if (!solid[k])
            c++;
    s[tid] = c;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0)
        part[blockIdx.x] = s[0];
}

// ── Host reduction (deterministic sequential) ──
static double host_reduce_3d(const double* d_partial, int nblocks) {
    std::vector<double> h(nblocks);
    cudaMemcpy(h.data(), d_partial, nblocks * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (double v : h)
        total += v;
    return total;
}

static int host_reduce_int_3d(const int* d_partial, int nblocks) {
    std::vector<int> h(nblocks);
    cudaMemcpy(h.data(), d_partial, nblocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int v : h)
        total += v;
    return total;
}

// ── Compute mean of v over INTERIOR fluid cells ──
static double compute_mean_3d(const double* d_v, const bool* d_solid, int nx, int ny, int nz,
                              int pitch, double* d_partial, int* d_count, int nblocks) {
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
    if (N_ >= N)
        return;
    free_buffers();
    cudaMalloc(&d_r, N * sizeof(double));
    cudaMalloc(&d_z, N * sizeof(double));
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_Ap, N * sizeof(double));
    cudaMalloc(&d_xt, N * sizeof(double)); // tile-native solution accumulator
    int max_blocks = (N + 255) / 256 + 1;
    cudaMalloc(&d_dot_buf, max_blocks * sizeof(double));
    cudaMalloc(&d_count_buf, max_blocks * sizeof(int));
    cudaMalloc(&d_scalar, sizeof(double));
    dot_buf_size_ = max_blocks;
    N_            = N;
}

void CudaPCG3D::free_buffers() {
    if (d_r)
        cudaFree(d_r);
    if (d_z)
        cudaFree(d_z);
    if (d_p)
        cudaFree(d_p);
    if (d_Ap)
        cudaFree(d_Ap);
    if (d_xt)
        cudaFree(d_xt);
    if (d_dot_buf)
        cudaFree(d_dot_buf);
    if (d_scalar)
        cudaFree(d_scalar);
    if (d_count_buf)
        cudaFree(d_count_buf);
    d_r = d_z = d_p = d_Ap = d_xt = d_dot_buf = nullptr;
    d_count_buf                               = nullptr;
    d_scalar                           = nullptr;
    dot_buf_size_                      = 0;
    N_                                 = 0;
    if (gf_N_ > 0) {
        gf_.free();
        if (d_rf)
            cudaFree(d_rf);
        if (d_zf)
            cudaFree(d_zf);
        d_rf = d_zf = nullptr;
        gf_N_       = 0;
    }
}

void CudaPCG3D::solve(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, nz = g.nz, pitch = g.pitch;
    int N = (nx + 2) * (ny + 2) * (nz + 2);
    ensure_buffers(N);
    dim3 block3d(8, 8, 8);
    dim3 grid3d((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    int nblocks1d = dot_buf_size_;

    precond_->setupLevels(g); // solid + Galerkin coeffs + §5.4 trimming — once per solve

    // ── Tile-native PCG: r/z/p/Ap/x live in the preconditioner's 8³ tile layout, so
    //    the V-cycle no longer scatters/gathers every iteration. Only the RHS (in)
    //    and the solution (out) cross the pitched↔tile boundary, once per solve. ──
    const long Nt    = precond_->level0_count(); // == nx*ny*nz (dims are ×8)
    const bool* tsol = precond_->level0_solid();
    int nbe          = (int)((Nt + 255) / 256); // one-thread-per-cell launches
    // r and z ALIAS the preconditioner's finest tile b and x → the V-cycle reads r
    // and writes z in place, so the solve never copies r/z anywhere (no scatter,
    // gather, or memcpy per iteration). Only rhs (in) and the solution (out) cross
    // the pitched↔tile boundary, once each per solve.
    double* rt = precond_->level0_b(); // r
    double* zt = precond_->level0_x(); // z
    precond_->to_tile(rhs, rt, g);     // r = scatter(rhs)
    auto tile_mean = [&](const double* v) -> double {
        sum_flat_kernel<<<nblocks1d, 256>>>(v, tsol, Nt, d_dot_buf);
        cudaDeviceSynchronize();
        double s = host_reduce_3d(d_dot_buf, nblocks1d);
        count_flat_kernel<<<nblocks1d, 256>>>(tsol, Nt, d_count_buf);
        cudaDeviceSynchronize();
        int c = host_reduce_int_3d(d_count_buf, nblocks1d);
        return c > 0 ? s / c : 0.0;
    };
    auto tile_dot = [&](const double* a, const double* b) -> double {
        dot_partial_kernel_3d<<<nblocks1d, 256>>>(a, b, tsol, (int)Nt, d_dot_buf);
        cudaDeviceSynchronize();
        return host_reduce_3d(d_dot_buf, nblocks1d);
    };
    double mr = tile_mean(rt); // r = -(r - mean)
    subtract_mean_flat_kernel<<<nbe, 256>>>(rt, mr, tsol, Nt);
    negate_flat_kernel<<<nbe, 256>>>(rt, tsol, Nt);

    // z = M^{-1} r (in place) ; subtract mean ; p = z ; x = 0
    precond_->vcycle_inplace();
    zt        = precond_->level0_x(); // up-leg ping-pong swaps the x buffer → re-fetch
    double mz = tile_mean(zt);
    subtract_mean_flat_kernel<<<nbe, 256>>>(zt, mz, tsol, Nt);
    cudaMemcpy(d_p, zt, Nt * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(d_xt, 0, Nt * sizeof(double));

    double rsold = tile_dot(rt, zt);
    if (rsold < 1e-30) {
        cudaMemset(p, 0, N * sizeof(double));
        return;
    }

    for (int k = 0; k < max_iter; k++) {
        precond_->matvec_tiled(d_p, d_Ap);
        double pAp = tile_dot(d_p, d_Ap);
        if (pAp < 1e-15)
            break;
        double alpha = rsold / pAp;

        axpy_flat_kernel<<<nbe, 256>>>(d_xt, d_p, alpha, tsol, Nt);  // x += alpha p
        axpy_flat_kernel<<<nbe, 256>>>(rt, d_Ap, -alpha, tsol, Nt);  // r -= alpha Ap
        CUDA_CHECK_3D(cudaDeviceSynchronize());

        double rsnew = tile_dot(rt, rt);
        if (std::sqrt(rsnew) < tol)
            break;

        precond_->vcycle_inplace(); // z = M^{-1} r (r in level0 b, z in level0 x)
        zt         = precond_->level0_x(); // re-fetch after the ping-pong swap
        double mz2 = tile_mean(zt);
        subtract_mean_flat_kernel<<<nbe, 256>>>(zt, mz2, tsol, Nt);

        double rz   = tile_dot(rt, zt);
        double beta = rz / rsold;
        rsold       = rz;
        xpby_flat_kernel<<<nbe, 256>>>(d_p, zt, beta, tsol, Nt); // p = z + beta p
        CUDA_CHECK_3D(cudaDeviceSynchronize());
    }
    // solution (tile) → pitched output
    precond_->from_tile(d_xt, p, g);
    CUDA_CHECK_3D(cudaDeviceSynchronize());
}

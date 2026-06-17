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

// ── FP32-input dot-product with FP64 accumulation (mixed: store fp32, accumulate fp64) ──
template <typename S>
__global__ void dot_partial_acc_kernel(const S* a, const S* b, const bool* solid, long N,
                                       double* d_partial) {
    __shared__ double sdata[256];
    int tid    = threadIdx.x;
    double sum = 0.0;
    for (long k = (long)blockIdx.x * blockDim.x + tid; k < N; k += (long)blockDim.x * gridDim.x) {
        if (!solid[k])
            sum += (double)a[k] * (double)b[k];
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

// ── FP32-input sum with FP64 accumulation ──
template <typename S>
__global__ void sum_flat_acc_kernel(const S* v, const bool* solid, long N, double* part) {
    __shared__ double s[256];
    int tid    = threadIdx.x;
    double sum = 0.0;
    for (long k = (long)blockIdx.x * blockDim.x + tid; k < N; k += (long)blockDim.x * gridDim.x)
        if (!solid[k])
            sum += (double)v[k];
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

// ── FP32 flat element-wise ops (tile layout, solid skipped) ──
__global__ void subtract_mean_flat_kernel_f(float* v, float mean, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        v[k] -= mean;
}
__global__ void xpby_flat_kernel_f(float* y, const float* x, float b, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        y[k] = x[k] + b * y[k];
}
// scatter rhs (FP64 pitched) → tile-layout FP32, via to_tile we already have a float pitched
__global__ void cast_d2f_flat(const double* d, float* f, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N)
        f[k] = (float)d[k];
}
__global__ void cast_f2d_flat(const float* f, double* d, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N)
        d[k] = (double)f[k];
}
// Mixed axpy: y(FP64) += a * x(FP32), accumulating in FP64 (residual recurrence).
__global__ void axpy_dmix_kernel(double* y, const float* x, double a, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        y[k] += a * (double)x[k];
}
// Mixed dot: a(FP64) · b(FP32), accumulating in FP64.
__global__ void dot_dmix_kernel(const double* a, const float* b, const bool* solid, long N,
                                double* part) {
    __shared__ double s[256];
    int tid    = threadIdx.x;
    double sum = 0.0;
    for (long k = (long)blockIdx.x * blockDim.x + tid; k < N; k += (long)blockDim.x * gridDim.x)
        if (!solid[k])
            sum += a[k] * (double)b[k];
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
// Cast FP64 tile → FP32 tile (feed the FP32 V-cycle's b buffer from the FP64 residual).
__global__ void cast_d2f_tile(const double* d, float* f, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N)
        f[k] = solid[k] ? 0.0f : (float)d[k];
}

// ═══════════════════════════════════════════════════════════════════════════
//  Device-resident scalar machinery (no host round-trip per PCG iteration).
//  Partial sums from the block-reduction kernels are reduced to a single device
//  scalar by reduce_partials_kernel; alpha/beta/mean are then computed by 1-thread
//  kernels reading device scalars; the axpy/xpby/submean ops consume those device
//  scalars directly. So a fixed-iteration solve issues every kernel back-to-back
//  with ZERO cudaMemcpy / cudaDeviceSynchronize inside the loop — exactly the
//  author's device-resident PCG that hits ~10.6 ms/step.
// ═══════════════════════════════════════════════════════════════════════════
// Reduce the FP64 partial-sum array (one entry per reduction block) to *out (device).
__global__ void reduce_partials_kernel(const double* part, int n, double* out) {
    __shared__ double s[256];
    int tid    = threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < n; i += 256)
        sum += part[i];
    s[tid] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st)
            s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0)
        *out = s[0];
}
// scalar device ops: o = n/d, o = -n/d, o = (cnt>0)? s/cnt : 0, o = i
__global__ void scd_div_k(double* o, const double* n, const double* d) {
    *o = (*d != 0.0) ? *n / *d : 0.0;
}
__global__ void scd_copy_k(double* o, const double* i) {
    *o = *i;
}
__global__ void scd_mean_k(double* o, const double* s, double cnt) {
    *o = (cnt > 0.0) ? *s / cnt : 0.0;
}
// beta(float) = rz/rsold (read from FP64 device scalars), written to a float scalar.
__global__ void scd_betaf_k(float* of, const double* rz, const double* rsold) {
    *of = (*rsold != 0.0) ? (float)(*rz / *rsold) : 0.0f;
}
// negate of an FP64 device scalar: o = -i
__global__ void scd_neg_k(double* o, const double* i) {
    *o = -(*i);
}
// FP64 tile y += (*a) x(FP32), reading α from a device scalar.
__global__ void axpy_dmix_dev_kernel(double* y, const float* x, const double* a, const bool* solid,
                                     long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        y[k] += (*a) * (double)x[k];
}
// FP32 tile p = z + (*b) p, reading β from a device float scalar.
__global__ void xpby_flat_dev_kernel_f(float* y, const float* x, const float* b, const bool* solid,
                                       long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        y[k] = x[k] + (*b) * y[k];
}
// FP64 tile v -= (*m), reading the mean from a device scalar.
__global__ void submean_flat_dev_kernel(double* v, const double* m, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        v[k] -= (*m);
}
// FP32 tile v -= (*mf), reading the mean from a device float scalar.
__global__ void submean_flat_dev_kernel_f(float* v, const float* mf, const bool* solid, long N) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !solid[k])
        v[k] -= (*mf);
}
// FP64 mean scalar → FP32 scalar (for the FP32 submean).
__global__ void scd_d2f_k(float* of, const double* i) {
    *of = (float)(*i);
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
    cudaMalloc(&d_sc, 8 * sizeof(double));  // device-resident scalar bank (FP64)
    cudaMalloc(&d_scf, 2 * sizeof(float));  // device-resident scalar bank (FP32)
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
    if (d_sc)
        cudaFree(d_sc);
    if (d_scf)
        cudaFree(d_scf);
    if (d_count_buf)
        cudaFree(d_count_buf);
    d_r = d_z = d_p = d_Ap = d_xt = d_dot_buf = nullptr;
    d_count_buf                               = nullptr;
    d_scalar                                  = nullptr;
    d_sc                                      = nullptr;
    d_scf                                     = nullptr;
    dot_buf_size_                             = 0;
    N_                                        = 0;
    if (gf_N_ > 0) {
        gf_.free();
        if (d_rf)
            cudaFree(d_rf);
        if (d_zf)
            cudaFree(d_zf);
        d_rf = d_zf = nullptr;
        gf_N_       = 0;
    }
    if (d_rf_pitch)
        cudaFree(d_rf_pitch);
    if (d_pf)
        cudaFree(d_pf);
    if (d_Apf)
        cudaFree(d_Apf);
    if (d_xtf)
        cudaFree(d_xtf);
    d_rf_pitch = d_pf = d_Apf = d_xtf = nullptr;
    f32_tile_N_  = 0;
    f32_pitch_N_ = 0;
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

// ═══════════════════════════════════════════════════════════════════════════
//  Mixed-precision tile-native PCG (the production projection path on consumer
//  GPUs, where FP64 ≈ 1/64 FP32). The expensive work — the warp V-cycle, the
//  matvec, and the FP32 search direction p / preconditioned z / Ap — runs in
//  FP32 in the 8³ tile layout. Only the accuracy-critical CG recurrence (the
//  residual r and solution x, updated r -= α·Ap / x += α·p) is kept in FP64,
//  which is what lets the solve reach the author's ~1e-3 |div| regime instead
//  of stagnating near the ~1e-2 floor a pure-FP32 outer loop hits. Dot/sum
//  reductions accumulate in FP64. RHS (FP64 pitched) is cast+scattered once;
//  the solution is gathered+cast back once. The FP64 `solve` stays available
//  (PCG_FP64=1) for the bit-for-bit CPU cross-check / high-accuracy needs.
// ═══════════════════════════════════════════════════════════════════════════
void CudaPCG3D::solve_f32_tile(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    int N = (nx + 2) * (ny + 2) * (nz + 2);
    ensure_buffers(N); // reuse d_dot_buf / d_count_buf (FP64 partials) + reduce helpers
    // Cap reduction blocks: kernels are grid-stride, so a small fixed block count
    // shrinks the per-dot D2H partial copy (the per-iteration host round-trip that
    // dominates once the V-cycle is FP32). 2048 partials sum deterministically.
    int nblocks1d = (N + 255) / 256 + 1;
    if (nblocks1d > 2048)
        nblocks1d = 2048;

    // ── Float grid mirroring g (uniform stencil from dx + solid). Set up once. ──
    if (gf_N_ < N) {
        if (gf_N_ > 0) {
            gf_.free();
            if (d_rf)
                cudaFree(d_rf);
            if (d_zf)
                cudaFree(d_zf);
            d_rf = d_zf = nullptr;
        }
        gf_.allocate(nx, ny, nz, (float)g.dx, (float)g.dy, (float)g.dz);
        gf_N_ = N;
    }
    if (f32_pitch_N_ < N) {
        if (d_rf_pitch)
            cudaFree(d_rf_pitch);
        cudaMalloc(&d_rf_pitch, N * sizeof(float));
        f32_pitch_N_ = N;
    }
    cudaMemcpy(gf_.solid, g.solid, N * sizeof(bool), cudaMemcpyDeviceToDevice);
    precond_f_->setupLevels(gf_);

    const long Nt    = precond_f_->level0_count();
    const bool* tsol = precond_f_->level0_solid();
    int nbe          = (int)((Nt + 255) / 256);
    if (f32_tile_N_ < Nt) {
        if (d_pf)
            cudaFree(d_pf);
        if (d_Apf)
            cudaFree(d_Apf);
        if (d_xtf)
            cudaFree(d_xtf);
        cudaMalloc(&d_pf, Nt * sizeof(float));
        cudaMalloc(&d_Apf, Nt * sizeof(float));
        cudaMalloc(&d_xtf, Nt * sizeof(float));
        f32_tile_N_ = Nt;
    }

    // ── Hybrid precision (the accuracy-critical recurrence stays FP64) ──
    //  r (residual) and x (solution) live in FP64 tile buffers — running the
    //  r -= α·Ap / x += α·p recurrence in FP64 is what lets the PCG escape the
    //  ~1e-2 stagnation floor of a pure-FP32 outer loop and reach the author's
    //  ~1e-3 regime in a few iters. p, z, Ap, the matvec, and the whole V-cycle
    //  run in FP32 (the expensive bandwidth/flops). d_r / d_xt (FP64, sized ≥Nt)
    //  are reused as the FP64 r / x tiles; precond_f_ level0 b/x are the FP32
    //  V-cycle in/out (also reused as scratch for the FP32 residual cast).
    int nbr    = (int)((N + 255) / 256);
    double* rt = d_r;  // residual (FP64 tile)
    double* xt = d_xt; // solution (FP64 tile)
    float* bf  = precond_f_->level0_b(); // FP32 V-cycle input
    float* zt  = precond_f_->level0_x(); // FP32 V-cycle output (preconditioned)

    // rhs (FP64 pitched) → FP32 pitched → FP32 tile (bf) → FP64 tile residual.
    cast_d2f_flat<<<nbr, 256>>>(rhs, d_rf_pitch, N);
    precond_f_->to_tile(d_rf_pitch, bf, gf_);
    cast_f2d_flat<<<nbe, 256>>>(bf, rt, Nt);

    // ── Device-resident scalar bank (no host round-trip inside the solve) ──
    //  d_sc: [0]=rsold [1]=pAp [2]=rsnew [3]=rz [6]=mean  ;  d_scf: [0]=beta [1]=mean(f)
    //  The fluid-cell count is constant for the whole solve, so it is the ONE
    //  scalar still read to the host (once, before the loop). Everything else —
    //  alpha/beta/mean and the r/x/p updates — stays on device, so a fixed-iter
    //  solve issues its kernels back-to-back with no cudaMemcpy / per-iter sync.
    count_flat_kernel<<<nblocks1d, 256>>>(tsol, Nt, d_count_buf);
    cudaDeviceSynchronize();
    double dcnt = (double)host_reduce_int_3d(d_count_buf, nblocks1d);

    // *d_sc[6] = mean(v over fluid) ; then v -= mean   (FP64, fully on device)
    auto dmean_d = [&](double* v) {
        sum_flat_acc_kernel<double><<<nblocks1d, 256>>>(v, tsol, Nt, d_dot_buf);
        reduce_partials_kernel<<<1, 256>>>(d_dot_buf, nblocks1d, d_sc + 6);
        scd_mean_k<<<1, 1>>>(d_sc + 6, d_sc + 6, dcnt);
        submean_flat_dev_kernel<<<nbe, 256>>>(v, d_sc + 6, tsol, Nt);
    };
    // *d_scf[1] = mean(v over fluid) ; then v -= mean   (FP32, fully on device)
    auto dmean_f = [&](float* v) {
        sum_flat_acc_kernel<float><<<nblocks1d, 256>>>(v, tsol, Nt, d_dot_buf);
        reduce_partials_kernel<<<1, 256>>>(d_dot_buf, nblocks1d, d_sc + 6);
        scd_mean_k<<<1, 1>>>(d_sc + 6, d_sc + 6, dcnt);
        scd_d2f_k<<<1, 1>>>(d_scf + 1, d_sc + 6);
        submean_flat_dev_kernel_f<<<nbe, 256>>>(v, d_scf + 1, tsol, Nt);
    };
    // reduce a freshly-launched partial array into device scalar d_sc[slot].
    auto ddot_dd = [&](const double* a, const double* b, int slot) {
        dot_partial_acc_kernel<double><<<nblocks1d, 256>>>(a, b, tsol, Nt, d_dot_buf);
        reduce_partials_kernel<<<1, 256>>>(d_dot_buf, nblocks1d, d_sc + slot);
    };
    auto ddot_ff = [&](const float* a, const float* b, int slot) {
        dot_partial_acc_kernel<float><<<nblocks1d, 256>>>(a, b, tsol, Nt, d_dot_buf);
        reduce_partials_kernel<<<1, 256>>>(d_dot_buf, nblocks1d, d_sc + slot);
    };
    auto ddot_df = [&](const double* a, const float* b, int slot) {
        dot_dmix_kernel<<<nblocks1d, 256>>>(a, b, tsol, Nt, d_dot_buf);
        reduce_partials_kernel<<<1, 256>>>(d_dot_buf, nblocks1d, d_sc + slot);
    };

    dmean_d(rt); // r -= mean(r)
    negate_flat_kernel<<<nbe, 256>>>(rt, tsol, Nt); // r = -(r - mean)

    // z = M⁻¹ r : cast r→FP32 b, run FP32 V-cycle, mean-remove z (FP32).
    cast_d2f_tile<<<nbe, 256>>>(rt, bf, tsol, Nt);
    precond_f_->vcycle_inplace_async();
    zt = precond_f_->level0_x();
    dmean_f(zt);
    cudaMemcpyAsync(d_pf, zt, Nt * sizeof(float), cudaMemcpyDeviceToDevice); // p = z (FP32)
    cudaMemsetAsync(xt, 0, Nt * sizeof(double));                             // x = 0 (FP64)

    ddot_df(rt, zt, 0); // rsold = (r,z)  → d_sc[0]
    last_iters   = max_iter;
    last_rel_res = 1.0;

    // ── Device-resident PCG with a PERIODIC convergence check (every CHECK_EVERY
    //    iters). Restores the early-exit the original solver had: a *fixed*-iteration
    //    FP32 CG loses conjugacy and starts DIVERGING on hard problems (delta-wing
    //    |div| spiked ~5e-1 when forced to run solve_iters=200), and over-iterating
    //    past convergence is also pure wasted time. Checking only every 20th iter
    //    keeps ~95% of the no-host-sync benefit. A short solve can't lose enough
    //    conjugacy to diverge, so CHECK_EVERY (20) is set above every "easy" solve's
    //    iteration count (collision: project@6, project_end@12) — those never reach a
    //    check, skip the r0² setup, and run fully device-resident with ZERO added host
    //    sync, exactly as before. Only the long hard solves (delta-wing 200/400) check. ──
    const int CHECK_EVERY = 20;
    double r0_sq = 0.0, tol_abs_sq = 0.0;
    if (max_iter > CHECK_EVERY) {
        ddot_dd(rt, rt, 2); // initial residual r0² (one host copy) for the convergence test
        cudaMemcpy(&r0_sq, d_sc + 2, sizeof(double), cudaMemcpyDeviceToHost);
        tol_abs_sq = (r0_sq > 0.0) ? r0_sq * tol * tol : tol * tol;
    }
    for (int k = 0; k < max_iter; k++) {
        precond_f_->matvec_tiled(d_pf, d_Apf); // Ap = A p   (FP32)
        ddot_ff(d_pf, d_Apf, 1);               // pAp → d_sc[1]
        scd_div_k<<<1, 1>>>(d_sc + 4, d_sc + 0, d_sc + 1); // alpha = rsold/pAp
        scd_neg_k<<<1, 1>>>(d_sc + 5, d_sc + 4);           // -alpha

        axpy_dmix_dev_kernel<<<nbe, 256>>>(xt, d_pf, d_sc + 4, tsol, Nt);   // x += α p
        axpy_dmix_dev_kernel<<<nbe, 256>>>(rt, d_Apf, d_sc + 5, tsol, Nt);  // r -= α Ap

        // True-residual convergence check. x and r are consistent here (x is the
        // current solution, r its residual), so breaking leaves x as the answer.
        if ((k % CHECK_EVERY) == (CHECK_EVERY - 1) && k + 1 < max_iter) {
            ddot_dd(rt, rt, 2);
            double rcheck = 0.0;
            cudaMemcpy(&rcheck, d_sc + 2, sizeof(double), cudaMemcpyDeviceToHost);
            if (rcheck < tol_abs_sq) {
                last_iters = k + 1;
                break;
            }
        }

        cast_d2f_tile<<<nbe, 256>>>(rt, bf, tsol, Nt); // FP32 V-cycle input
        precond_f_->vcycle_inplace_async();
        zt = precond_f_->level0_x();
        dmean_f(zt);

        ddot_df(rt, zt, 3);                                // rz → d_sc[3]
        scd_betaf_k<<<1, 1>>>(d_scf + 0, d_sc + 3, d_sc + 0); // beta(f) = rz/rsold
        scd_copy_k<<<1, 1>>>(d_sc + 0, d_sc + 3);          // rsold = rz
        xpby_flat_dev_kernel_f<<<nbe, 256>>>(d_pf, zt, d_scf + 0, tsol, Nt); // p = z + β p
    }
    // Final true residual for last_rel_res reporting (one host sync).
    ddot_dd(rt, rt, 2);
    double rsn = 0.0;
    cudaMemcpy(&rsn, d_sc + 2, sizeof(double), cudaMemcpyDeviceToHost);
    last_rel_res = std::sqrt(rsn / (r0_sq > 0.0 ? r0_sq : 1.0));

    // solution (FP64 tile) → FP32 tile → pitched FP32 → FP64 output. The output
    // cast through FP32 is harmless: it feeds the velocity correction only.
    cast_d2f_tile<<<nbe, 256>>>(xt, d_xtf, tsol, Nt);
    precond_f_->from_tile(d_xtf, d_rf_pitch, gf_);
    cast_f2d_flat<<<nbr, 256>>>(d_rf_pitch, p, N);
    CUDA_CHECK_3D(cudaDeviceSynchronize());
}

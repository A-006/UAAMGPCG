#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK_3D(call)                                                                        \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
            return;                                                                                \
        }                                                                                          \
    } while (0)

constexpr int TILE_DIM_3D = 8;

// ── Templated grid (supports float and double) ──
template <typename T> struct CudaGrid3DT_ {
    int nx, ny, nz;
    T dx, dy, dz, idx2, idy2, idz2, diag;
    T *x = nullptr, *b = nullptr;
    bool* solid = nullptr;
    int pitch   = 0;
    // 8x8x8 tile-contiguous (SoA) layout metadata (Sun et al. SIGGRAPH 2025 §5.3).
    // tnx/tny/tnz = number of 8^3 tiles per axis; num_tiles*512 = storage length.
    // tiled==false → classic pitched-linear layout (default, used by callers/PCG).
    int tnx = 0, tny = 0, tnz = 0;
    long num_tiles = 0;
    bool tiled     = false;
    void allocate(int nx_, int ny_, int nz_, T dx_, T dy_, T dz_);
    void allocate_tiled(int nx_, int ny_, int nz_, T dx_, T dy_, T dz_);
    long flat_size() const {
        return tiled ? num_tiles * 512 : (long)(nx + 2) * (ny + 2) * (nz + 2);
    }
    void free();
};

template <typename T>
void CudaGrid3DT_<T>::allocate(int nx_, int ny_, int nz_, T dx_, T dy_, T dz_) {
    nx    = nx_;
    ny    = ny_;
    nz    = nz_;
    dx    = dx_;
    dy    = dy_;
    dz    = dz_;
    idx2  = T(1.0) / (dx * dx);
    idy2  = T(1.0) / (dy * dy);
    idz2  = T(1.0) / (dz * dz);
    diag  = T(2.0) * (idx2 + idy2 + idz2);
    int N = (nx + 2) * (ny + 2) * (nz + 2);
    cudaMalloc(&x, N * sizeof(T));
    cudaMemset(x, 0, N * sizeof(T));
    cudaMalloc(&b, N * sizeof(T));
    cudaMemset(b, 0, N * sizeof(T));
    cudaMalloc(&solid, N * sizeof(bool));
    cudaMemset(solid, 0, N * sizeof(bool));
    pitch = nx + 2;
    tiled = false;
}

// Tile-contiguous (SoA) allocation: interior cells (1..n) map to 8^3 tiles, a
// voxel (i,j,k) lives at tile_idx*512 + voxel_idx (see dev_idx3d). Each channel
// is its own contiguous array of num_tiles*512 elements → far better L2 locality
// for the bandwidth-bound RBGS smoother on the 3090's small L2. No ghost cells.
template <typename T>
void CudaGrid3DT_<T>::allocate_tiled(int nx_, int ny_, int nz_, T dx_, T dy_, T dz_) {
    nx        = nx_;
    ny        = ny_;
    nz        = nz_;
    dx        = dx_;
    dy        = dy_;
    dz        = dz_;
    idx2      = T(1.0) / (dx * dx);
    idy2      = T(1.0) / (dy * dy);
    idz2      = T(1.0) / (dz * dz);
    diag      = T(2.0) * (idx2 + idy2 + idz2);
    tnx       = (nx + 7) / 8;
    tny       = (ny + 7) / 8;
    tnz       = (nz + 7) / 8;
    num_tiles = (long)tnx * tny * tnz;
    long N    = num_tiles * 512;
    cudaMalloc(&x, N * sizeof(T));
    cudaMemset(x, 0, N * sizeof(T));
    cudaMalloc(&b, N * sizeof(T));
    cudaMemset(b, 0, N * sizeof(T));
    cudaMalloc(&solid, N * sizeof(bool));
    cudaMemset(solid, 0, N * sizeof(bool));
    pitch = nx + 2;
    tiled = true;
}

template <typename T> void CudaGrid3DT_<T>::free() {
    if (x)
        cudaFree(x);
    if (b)
        cudaFree(b);
    if (solid)
        cudaFree(solid);
    x     = nullptr;
    b     = nullptr;
    solid = nullptr;
}

// CudaGrid3D is the double instantiation (kept as the canonical name used across
// the double solver/tests); CudaGrid3Df is the float instantiation.
using CudaGrid3D  = CudaGrid3DT_<double>;
using CudaGrid3Df = CudaGrid3DT_<float>;

#pragma once
#include <cuda_runtime.h>
#include <cstdio>

// ── Error checking ──
#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(e));       \
        return;                                                  \
    }                                                            \
} while(0)

#define CUDA_CHECK_V(call, val) do {                             \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(e));       \
        return (val);                                            \
    }                                                            \
} while(0)

// ── Tile size (paper: 8x8x8 for 3D → 8x8 for 2D) ──
constexpr int TILE_DIM = 8;

// ── GPU grid data: SOA layout (Section 5.2) ──
// Separate channels: x (pressure), b (rhs), solid mask
// Matrix represented implicitly via grid dimensions dx, dy
struct CudaGrid {
    int nx, ny;
    double dx, dy;
    double idx2, idy2, diag;  // cached stencil coefficients

    double *x = nullptr;       // solution / correction
    double *b = nullptr;       // right-hand side
    bool   *solid = nullptr;   // DoF mask (false = fluid, true = solid)

    int nfluid = 0;            // number of fluid cells
    int pitch = 0;             // padded row size in elements (nx+2)
    size_t pitch_bytes = 0;

    void allocate(int nx_, int ny_, double dx_, double dy_);
    void free();
};

inline void CudaGrid::allocate(int nx_, int ny_, double dx_, double dy_) {
    nx = nx_; ny = ny_; dx = dx_; dy = dy_;
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy); diag = 2.0*(idx2+idy2);
    int N = (nx+2)*(ny+2);
    cudaMalloc(&x, N * sizeof(double));     cudaMemset(x, 0, N * sizeof(double));
    cudaMalloc(&b, N * sizeof(double));     cudaMemset(b, 0, N * sizeof(double));
    cudaMalloc(&solid, N * sizeof(bool));   cudaMemset(solid, 0, N * sizeof(bool));
    nfluid = 0;
    pitch = nx + 2;
    pitch_bytes = pitch * sizeof(double);
}

inline void CudaGrid::free() {
    if (x)     cudaFree(x);
    if (b)     cudaFree(b);
    if (solid) cudaFree(solid);
    x = b = nullptr; solid = nullptr;
}

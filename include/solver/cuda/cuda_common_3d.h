#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK_3D(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return; \
    } \
} while(0)

constexpr int TILE_DIM_3D = 8;

struct CudaGrid3D {
    int nx, ny, nz;
    double dx, dy, dz;
    double idx2, idy2, idz2, diag;

    double *x = nullptr;    // solution/correction
    double *b = nullptr;    // RHS
    bool   *solid = nullptr;
    int pitch = 0;          // stride = nx+2

    void allocate(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_);
    void free();
};

inline void CudaGrid3D::allocate(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_) {
    nx=nx_; ny=ny_; nz=nz_; dx=dx_; dy=dy_; dz=dz_;
    idx2=1.0/(dx*dx); idy2=1.0/(dy*dy); idz2=1.0/(dz*dz);
    diag=2.0*(idx2+idy2+idz2);
    int N=(nx+2)*(ny+2)*(nz+2);
    cudaMalloc(&x, N*sizeof(double));     cudaMemset(x,0,N*sizeof(double));
    cudaMalloc(&b, N*sizeof(double));     cudaMemset(b,0,N*sizeof(double));
    cudaMalloc(&solid, N*sizeof(bool));   cudaMemset(solid,0,N*sizeof(bool));
    pitch=nx+2;
}

inline void CudaGrid3D::free() {
    if(x) cudaFree(x);
    if(b) cudaFree(b);
    if(solid) cudaFree(solid);
    x=nullptr; b=nullptr; solid=nullptr;
}

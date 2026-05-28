/**
 * @file test_fp32_vcycle.cu
 * @brief FP64 vs FP32 Galerkin V-cycle timing (precision optimization, §5 / FP32).
 *
 * Times the templated CudaUAAMGPreconditioner3DT<T> V-cycle for T=double and
 * T=float at several grids. FP32 halves global-memory traffic.
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"
#include <cstdio>
#include <vector>
#include <chrono>

template<typename T>
static double bench_vcycle(int nx, int ny, int nz) {
    CudaGrid3DT_<T> g;
    g.allocate(nx, ny, nz, T(1.0/nx), T(1.0/ny), T(1.0/nz));
    int N = (nx+2)*(ny+2)*(nz+2);
    std::vector<T> hr(N);
    for (int i = 0; i < N; i++) hr[i] = T((i % 97) * 0.01 - 0.5);
    T *d_r, *d_z;
    cudaMalloc(&d_r, N*sizeof(T)); cudaMalloc(&d_z, N*sizeof(T));
    cudaMemcpy(d_r, hr.data(), N*sizeof(T), cudaMemcpyHostToDevice);

    CudaUAAMGPreconditioner3DT<T> p;
    p.setupLevels(g);
    for (int w = 0; w < 5; w++) p.vcycle_apply(g, d_r, d_z);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < 20; r++) p.vcycle_apply(g, d_r, d_z);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaFree(d_r); cudaFree(d_z); g.free();
    return std::chrono::duration<double>(t1-t0).count()*1000.0/20.0;
}

int main() {
    printf("=== FP64 vs FP32 Galerkin V-cycle (RTX 3090, with §5.4 trimming) ===\n");
    int grids[][3] = {{64,32,32},{128,64,64},{256,128,128},{256,256,256}};
    for (auto& G : grids) {
        double d = bench_vcycle<double>(G[0],G[1],G[2]);
        double f = bench_vcycle<float >(G[0],G[1],G[2]);
        printf("  %3dx%3dx%3d : FP64 %.3f ms   FP32 %.3f ms   speedup %.2fx\n",
               G[0],G[1],G[2], d, f, d/f);
    }
    return 0;
}

/**
 * @file bench_vcycle256.cu
 * @brief Minimal single-size FP32 V-cycle driver for clean ncu profiling.
 * Usage: bench_vcycle256 [nx] [ny] [nz] [reps]   default 256 256 256 10
 */
#include "solver/cuda/preconditioner/cuda_uaamg_preconditioner_3d.h"
#include <chrono>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int nx   = argc > 1 ? atoi(argv[1]) : 256;
    int ny   = argc > 2 ? atoi(argv[2]) : 256;
    int nz   = argc > 3 ? atoi(argv[3]) : 256;
    int reps = argc > 4 ? atoi(argv[4]) : 10;

    CudaGrid3DT_<float> g;
    g.allocate(nx, ny, nz, 1.0f / nx, 1.0f / ny, 1.0f / nz);
    int N = (nx + 2) * (ny + 2) * (nz + 2);
    std::vector<float> hr(N);
    for (int i = 0; i < N; i++)
        hr[i] = (i % 97) * 0.01f - 0.5f;
    float *d_r, *d_z;
    cudaMalloc(&d_r, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMemcpy(d_r, hr.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    CudaUAAMGPreconditioner3DT<float> p;
    p.setupLevels(g);
    for (int w = 0; w < 3; w++) // warmup
        p.vcycle_apply(g, d_r, d_z);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++)
        p.vcycle_apply(g, d_r, d_z);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("V-cycle %dx%dx%d FP32: %.3f ms/cycle (%d reps)\n", nx, ny, nz,
           std::chrono::duration<double>(t1 - t0).count() * 1000.0 / reps, reps);
    cudaFree(d_r);
    cudaFree(d_z);
    g.free();
    return 0;
}

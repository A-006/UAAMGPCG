/**
 * @file bench_vcycle_fresh.cu
 * @brief Fresh CUDA-event-timed single FP32 V-cycle microbenchmark @256^3.
 *
 * Uniform fully-fluid domain (no solids) -> exercises the §5.4 trivial/trim
 * fast path, apples-to-apples with the authors' AMGPCG (FP32, all-DOF).
 *
 * Measures BOTH:
 *   (1) vcycle_inplace()  -- the REAL per-PCG-iteration cost the production
 *       solve pays (b aliases r, x aliases z; no per-call scatter/gather).
 *       This is the apples-to-apples figure vs the authors' VcycleDotAsync.
 *   (2) vcycle_apply()    -- legacy path = scatter + vcycle_inplace + gather.
 *       Reported for continuity with the historical 5.78 ms "template path".
 *
 * Warm up, then time N V-cycles with cudaEvents, device-synced. Reports
 * mean/min/stddev.
 *
 * Usage: bench_vcycle_fresh [nx] [ny] [nz] [reps] [warmup]
 *        default 256 256 256 100 10
 */
#include "solver/cuda/preconditioner/cuda_uaamg_preconditioner_3d.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

template <typename F>
static void time_loop(const char* label, int reps, F&& once) {
    // warmup already done by caller
    std::vector<float> samples(reps);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(e0, 0);
        once();
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&samples[r], e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    double sum = 0, mn = 1e30, mx = 0;
    for (float s : samples) {
        sum += s;
        if (s < mn) mn = s;
        if (s > mx) mx = s;
    }
    double mean = sum / reps;
    double var = 0;
    for (float s : samples) var += (s - mean) * (s - mean);
    double sd = std::sqrt(var / reps);
    printf("  %-18s mean %.4f ms   min %.4f   max %.4f   stddev %.4f  (%d reps)\n",
           label, mean, mn, mx, sd, reps);
}

int main(int argc, char** argv) {
    int nx     = argc > 1 ? atoi(argv[1]) : 256;
    int ny     = argc > 2 ? atoi(argv[2]) : 256;
    int nz     = argc > 3 ? atoi(argv[3]) : 256;
    int reps   = argc > 4 ? atoi(argv[4]) : 100;
    int warmup = argc > 5 ? atoi(argv[5]) : 10;

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s\n", prop.name);
    printf("OUR UAAMG V-cycle  %dx%dx%d FP32  uniform fully-fluid (no solids)\n", nx, ny, nz);

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

    // ---- (1) vcycle_inplace: real per-iteration path (no scatter/gather) ----
    // Seed the tile-native b once (=r); x is written each call.
    p.to_tile(d_r, p.level0_b(), g);
    for (int w = 0; w < warmup; w++)
        p.vcycle_inplace();
    cudaDeviceSynchronize();
    time_loop("vcycle_inplace", reps, [&] { p.vcycle_inplace(); });

    // ---- (2) vcycle_apply: legacy path = scatter + vcycle_inplace + gather ----
    for (int w = 0; w < warmup; w++)
        p.vcycle_apply(g, d_r, d_z);
    cudaDeviceSynchronize();
    time_loop("vcycle_apply", reps, [&] { p.vcycle_apply(g, d_r, d_z); });

    cudaError_t err = cudaGetLastError();
    printf("  [%s]\n", cudaGetErrorString(err));

    cudaFree(d_r);
    cudaFree(d_z);
    g.free();
    return 0;
}

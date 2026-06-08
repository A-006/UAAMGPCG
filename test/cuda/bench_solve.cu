// Time a full FP64 PCG solve (the path the 3D LFM pressure projection uses) at a
// realistic size — measures the per-iteration cost incl. preconditioner V-cycle.
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/cuda/cuda_common_3d.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int nx = argc > 1 ? atoi(argv[1]) : 256;
    int ny = argc > 2 ? atoi(argv[2]) : 256;
    int nz = argc > 3 ? atoi(argv[3]) : 256;
    int iters = argc > 4 ? atoi(argv[4]) : 20;
    int N = (nx + 2) * (ny + 2) * (nz + 2);
    std::vector<double> h_rhs(N, 0.0);
    srand(1);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                int id    = i + j * (nx + 2) + k * (nx + 2) * (ny + 2);
                h_rhs[id] = (rand() % 1000) / 1000.0 - 0.5;
            }
    CudaGrid3D g;
    g.allocate(nx, ny, nz, 1.0 / nx, 1.0 / nx, 1.0 / nx); // cubic cells
    cudaMemset(g.solid, 0, N * sizeof(bool));
    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_rhs, N * sizeof(double));
    cudaMemcpy(d_rhs, h_rhs.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    CudaPCG3D pcg;
    for (int w = 0; w < 3; w++) { cudaMemset(d_p, 0, N * sizeof(double)); pcg.solve(g, d_p, d_rhs, iters, 0.0); }
    cudaDeviceSynchronize();
    const int reps = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++) { cudaMemset(d_p, 0, N * sizeof(double)); pcg.solve(g, d_p, d_rhs, iters, 0.0); }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / reps;
    printf("FP64 solve %dx%dx%d, %d iters: %.2f ms  (%.3f ms/iter)\n", nx, ny, nz, iters, ms,
           ms / iters);
    return 0;
}

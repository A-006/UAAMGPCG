// Fast convergence oracle (correct A, matches test_paper_bench). PCG solve at a
// fused-path size; reports rel residual per iteration. No CPU reference.
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/cuda/cuda_common_3d.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static double residual_l2(const double* p, const double* rhs, const bool* solid, int nx, int ny,
                          int nz, double dx, double dy, double dz) {
    int stride = nx + 2, sy = ny + 2;
    double idx2 = 1.0 / (dx * dx), idy2 = 1.0 / (dy * dy), idz2 = 1.0 / (dz * dz),
           diag = 2.0 * (idx2 + idy2 + idz2);
    double sum = 0;
    int cnt    = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                int id = i + j * stride + k * stride * sy;
                if (!solid[id]) {
                    sum += rhs[id];
                    cnt++;
                }
            }
    double mean = cnt > 0 ? sum / cnt : 0, s2 = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                int id = i + j * stride + k * stride * sy;
                if (solid[id])
                    continue;
                double pC = p[id];
                double pL = (i > 1 && !solid[id - 1]) ? p[id - 1] : pC;
                double pR = (i < nx && !solid[id + 1]) ? p[id + 1] : pC;
                double pB = (j > 1 && !solid[id - stride]) ? p[id - stride] : pC;
                double pT = (j < ny && !solid[id + stride]) ? p[id + stride] : pC;
                double pF = (k > 1 && !solid[id - stride * sy]) ? p[id - stride * sy] : pC;
                double pK = (k < nz && !solid[id + stride * sy]) ? p[id + stride * sy] : pC;
                double Ax = diag * pC - (pL + pR) * idx2 - (pB + pT) * idy2 - (pF + pK) * idz2;
                double r  = -(rhs[id] - mean) - Ax;
                s2 += r * r;
            }
    return std::sqrt(s2);
}

int main(int argc, char** argv) {
    int nx = argc > 1 ? atoi(argv[1]) : 256;
    int ny = argc > 2 ? atoi(argv[2]) : 128;
    int nz = argc > 3 ? atoi(argv[3]) : 128;
    int N  = (nx + 2) * (ny + 2) * (nz + 2);
    std::vector<double> h_rhs(N, 0.0);
    std::vector<char> h_solid(N, 0);
    srand(1234);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                int id    = i + j * (nx + 2) + k * (nx + 2) * (ny + 2);
                double lo = (double)i / nx + (double)j / ny + (double)k / nz;
                double hi = (rand() % 1000) / 1000.0 - 0.5;
                h_rhs[id] = lo * 0.5 + hi * 0.5;
            }
    CudaGrid3D g;
    g.allocate(nx, ny, nz, 1.0 / nx, 1.0 / ny, 1.0 / nz);
    cudaMemcpy(g.solid, h_solid.data(), N * sizeof(bool), cudaMemcpyHostToDevice);
    std::vector<char> sc(N);
    cudaMemcpy(sc.data(), g.solid, N * sizeof(bool), cudaMemcpyDeviceToHost);
    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_rhs, N * sizeof(double));
    double r0 = residual_l2(std::vector<double>(N, 0.0).data(), h_rhs.data(), (bool*)sc.data(), nx,
                            ny, nz, 1.0 / nx, 1.0 / ny, 1.0 / nz);
    printf("Grid %dx%dx%d  ||r0||=%.4e\n", nx, ny, nz, r0);
    for (int iters : {1, 5, 10, 20, 40}) {
        cudaMemset(d_p, 0, N * sizeof(double));
        cudaMemcpy(d_rhs, h_rhs.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        CudaPCG3D pcg;
        pcg.solve(g, d_p, d_rhs, iters, 1e-12);
        std::vector<double> gp(N);
        cudaMemcpy(gp.data(), d_p, N * sizeof(double), cudaMemcpyDeviceToHost);
        double r = residual_l2(gp.data(), h_rhs.data(), (bool*)sc.data(), nx, ny, nz, 1.0 / nx,
                               1.0 / ny, 1.0 / nz);
        printf("  iter=%2d  rel=%.4e\n", iters, r / r0);
        fflush(stdout);
    }
    return 0;
}

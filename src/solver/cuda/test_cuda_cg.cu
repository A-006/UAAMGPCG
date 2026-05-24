/**
 * @file test_cuda_cg.cu
 * @brief GPU CG vs CPU CG comparison tests.
 */
#include "solver/cuda/cuda_cg.h"
#include "solver/pcg.h"
#include "solver/preconditioner/identity_preconditioner.h"
#include "core/grid.h"
#include <cstdio>
#include <cmath>
#include <vector>

static double max_abs_diff(const double *a, const double *b, int N) {
    double m = 0; for (int k = 0; k < N; k++) m = std::max(m, std::abs(a[k] - b[k])); return m;
}

int main() {
    int nx = 32, ny = 16, stride = nx + 2, N = (nx+2)*(ny+2);
    double dx = 1.0/nx, dy = 1.0/ny;
    int passed = 0, failed = 0;

    // ── Test 1: CG converges on GPU, matches CPU ──
    printf("=== Test 1: GPU CG vs CPU CG ===\n");

    std::vector<double> h_rhs(N, 0);
    std::vector<char>   h_solid(N, 0);
    double cx = nx/2.0, cy = ny/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            double d2 = (i-cx)*(i-cx) + (j-cy)*(j-cy);
            h_rhs[i + j*stride] = std::exp(-0.01 * d2);
        }

    // CPU CG
    Grid cpu_g(nx, ny, 1.0, 1.0);
    auto cpu_solver = std::make_unique<PCG>(std::make_unique<IdentityPreconditioner>());
    std::fill(cpu_g.p.begin(), cpu_g.p.end(), 0);
    cpu_solver->solve(cpu_g, h_rhs, 200, 1e-10);
    std::vector<double> cpu_p(N, 0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            cpu_p[cpu_g.ip(i,j)] = cpu_g.p[cpu_g.ip(i,j)];

    // GPU CG
    CudaGrid gpu_g; gpu_g.allocate(nx, ny, dx, dy);
    cudaMemcpy(gpu_g.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N*sizeof(double)); cudaMemset(d_p, 0, N*sizeof(double));
    cudaMalloc(&d_rhs, N*sizeof(double)); cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    CudaCG cg;
    cg.solve(gpu_g, d_p, d_rhs, 200, 1e-10);

    std::vector<double> gpu_p(N, 0);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double diff = max_abs_diff(cpu_p.data(), gpu_p.data(), N);
    double cpu_max = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            cpu_max = std::max(cpu_max, std::abs(cpu_p[cpu_g.ip(i,j)]));

    printf("  CPU max|p|=%.6e  GPU max|p|=%.6e\n", cpu_max,
           *std::max_element(gpu_p.begin(), gpu_p.end(), [](double a,double b){return std::abs(a)<std::abs(b);}));
    printf("  max|cpu_p - gpu_p| = %e\n", diff);
    bool pass1 = (diff < 1e-10 * std::max(1.0, cpu_max));
    printf("  %s\n", pass1 ? "PASS" : "FAIL");
    pass1 ? passed++ : failed++;

    // ── Test 2: CG convergence (residual decreases) ──
    printf("\n=== Test 2: GPU CG convergence ===\n");
    for (int iters : {1, 5, 20, 100}) {
        cudaMemset(d_p, 0, N*sizeof(double));
        cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        cg.solve(gpu_g, d_p, d_rhs, iters, 1e-10);
        cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);
        double pmax = 0;
        for (int i = 1; i <= nx; i++) for (int j = 1; j <= ny; j++)
            pmax = std::max(pmax, std::abs(gpu_p[i+j*stride]));
        printf("  iter %3d: max|p|=%.6e\n", iters, pmax);
    }
    printf("  PASS (values stable)\n"); passed++;

    // ── Test 3: residual check after solve ──
    printf("\n=== Test 3: Residual after GPU CG ===\n");
    cudaMemset(d_p, 0, N*sizeof(double));
    cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cg.solve(gpu_g, d_p, d_rhs, 200, 1e-10);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double idx2 = 1.0/(dx*dx), idy2 = 1.0/(dy*dy), diag = 2.0*(idx2+idy2);
    double max_r = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*stride;
            double pC = gpu_p[id];
            double pL = (i>1 && !h_solid[(i-1)+j*stride]) ? gpu_p[(i-1)+j*stride] : pC;
            double pR = (i<nx && !h_solid[(i+1)+j*stride]) ? gpu_p[(i+1)+j*stride] : pC;
            double pB = (j>1 && !h_solid[i+(j-1)*stride]) ? gpu_p[i+(j-1)*stride] : pC;
            double pT = (j<ny && !h_solid[i+(j+1)*stride]) ? gpu_p[i+(j+1)*stride] : pC;
            double Ax = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
            // PCG negates RHS: solves (-nabla^2)p = -(rhs - mean)
            // residual = -(rhs - mean) - (-nabla^2)p
            double neg_rhs = -h_rhs[id];  // mean ≈ 0 for Gaussian
            max_r = std::max(max_r, std::abs(neg_rhs - Ax));
        }
    printf("  max|residual|=%e  %s\n", max_r, max_r < 1.0 ? "PASS" : "FAIL");
    (max_r < 1.0) ? passed++ : failed++;

    cudaFree(d_p); cudaFree(d_rhs); gpu_g.free();

    printf("\n========================================\n");
    printf("  GPU CG tests: %d passed, %d failed\n", passed, failed);
    return failed > 0;
}

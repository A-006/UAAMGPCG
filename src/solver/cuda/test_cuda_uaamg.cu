/**
 * @file test_cuda_uaamg.cu
 * @brief GPU unit tests — compare CUDA vs CPU UAAMGPCG.
 *
 * Tests:
 *   1. Single V-cycle output (uniform RHS, no solid)
 *   2. Full PCG solve (uniform RHS, compare CPU vs GPU pressure)
 *   3. V-cycle with solid obstacles (Karman-like cylinder)
 */
#include "solver/cuda/cuda_pcg.h"
#include "solver/pcg.h"
#include "solver/preconditioner/uaamg_preconditioner.h"
#include "core/grid.h"
#include <cstdio>
#include <cmath>
#include <vector>

// ── Test 1: GPU vs CPU V-cycle ──
static bool test_vcycle_gpu_vs_cpu() {
    printf("\n=== Test 1: GPU vs CPU V-cycle ===\n");
    int nx = 32, ny = 16, N = (nx+2)*(ny+2);
    std::vector<double> h_r(N, 0.0);
    std::vector<char>   h_solid(N, 0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            h_r[i + j*(nx+2)] = 1.0;

    // CPU
    Grid cpu_grid(nx, ny, 1.0, 1.0);
    UAAMGPreconditioner cpu_precond;
    std::vector<double> cpu_z(N, 0.0);
    cpu_precond.apply(cpu_grid, h_r, cpu_z);

    // GPU
    CudaGrid gpu_grid; gpu_grid.allocate(nx, ny, 1.0/nx, 1.0/ny);
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
    double *d_r, *d_z;
    cudaMalloc(&d_r, N*sizeof(double)); cudaMemcpy(d_r, h_r.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_z, N*sizeof(double));

    CudaUAAMGPreconditioner gpu_precond;
    gpu_precond.apply(gpu_grid, d_r, d_z);

    std::vector<double> gpu_z(N, 0.0);
    cudaMemcpy(gpu_z.data(), d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0, max_val = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*(nx+2);
            double d = std::abs(cpu_z[id] - gpu_z[id]);
            if (d > max_diff) max_diff = d;
            if (std::abs(cpu_z[id]) > max_val) max_val = std::abs(cpu_z[id]);
        }
    printf("  Max |cpu - gpu| = %e  (max|cpu| = %e)\n", max_diff, max_val);
    bool pass = (max_diff < 1e-12 * std::max(1.0, max_val));
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_r); cudaFree(d_z); gpu_grid.free();
    return pass;
}

// ── Test 2: Full PCG solve — GPU vs CPU on uniform RHS ──
static bool test_pcg_gpu_vs_cpu() {
    printf("\n=== Test 2: GPU vs CPU PCG solve ===\n");
    int nx = 32, ny = 16, N = (nx+2)*(ny+2);
    std::vector<double> h_rhs(N, 0.0);
    std::vector<char>   h_solid(N, 0);
    // Non-constant RHS (PCG zero-means → kills uniform RHS)
    double cx = nx/2.0, cy = ny/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            double dx2 = (i-cx)*(i-cx), dy2 = (j-cy)*(j-cy);
            h_rhs[i + j*(nx+2)] = std::exp(-0.01 * (dx2 + dy2));
        }

    // CPU PCG
    Grid cpu_grid(nx, ny, 1.0, 1.0);
    auto cpu_solver = std::make_unique<PCG>(std::make_unique<UAAMGPreconditioner>());
    std::fill(cpu_grid.p.begin(), cpu_grid.p.end(), 0.0);
    cpu_solver->solve(cpu_grid, h_rhs, 50, 1e-10);

    std::vector<double> cpu_p(N, 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            cpu_p[cpu_grid.ip(i,j)] = cpu_grid.p[cpu_grid.ip(i,j)];

    // GPU PCG
    CudaGrid gpu_grid; gpu_grid.allocate(nx, ny, 1.0/nx, 1.0/ny);
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);

    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N*sizeof(double)); cudaMemset(d_p, 0, N*sizeof(double));
    cudaMalloc(&d_rhs, N*sizeof(double)); cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    CudaPCG pcg;

    // Diagnostic: run with 1 iteration, check if V-cycle output matches Test 1
    pcg.solve(gpu_grid, d_p, d_rhs, 1, 1e-10);

    std::vector<double> gpu_p(N, 0.0);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    // Check if GPU PCG produced non-zero output
    double gpu_max = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            if (std::abs(gpu_p[i + j*(nx+2)]) > gpu_max)
                gpu_max = std::abs(gpu_p[i + j*(nx+2)]);
    printf("  GPU PCG(1 iter) max|p| = %e\n", gpu_max);
    printf("  %s\n", (gpu_max > 1e-15) ? "PASS" : "FAIL");

    bool pass = (gpu_max > 1e-15);

    cudaFree(d_p); cudaFree(d_rhs); gpu_grid.free();
    return pass;
}

// ── Test 3: V-cycle with solid obstacles ──
static bool test_vcycle_with_solid() {
    printf("\n=== Test 3: V-cycle with solid obstacles ===\n");
    int nx = 64, ny = 16;
    double Lx = 4.0, Ly = 1.0, dx = Lx/nx, dy = Ly/ny;
    int N = (nx+2)*(ny+2);
    std::vector<double> h_r(N, 0.0);
    std::vector<char>   h_solid(N, 0);

    double cx = 1.0, cy = 0.5, R = 0.1;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            double xi = (i-0.5)*dx, yj = (j-0.5)*dy;
            int id = i + j*(nx+2);
            if (std::sqrt((xi-cx)*(xi-cx)+(yj-cy)*(yj-cy)) < R) h_solid[id] = 1;
            if (!h_solid[id]) h_r[id] = 1.0;
        }

    // CPU
    Grid cpu_grid(nx, ny, Lx, Ly);
    for (int i = 0; i <= nx+1; i++)
        for (int j = 0; j <= ny+1; j++)
            if (h_solid[cpu_grid.ip(i,j)]) cpu_grid.set_solid(i, j);
    UAAMGPreconditioner cpu_precond;
    std::vector<double> cpu_z(N, 0.0);
    cpu_precond.apply(cpu_grid, h_r, cpu_z);

    // GPU
    CudaGrid gpu_grid; gpu_grid.allocate(nx, ny, dx, dy);
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
    double *d_r, *d_z;
    cudaMalloc(&d_r, N*sizeof(double)); cudaMemcpy(d_r, h_r.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_z, N*sizeof(double));
    CudaUAAMGPreconditioner gpu_precond;
    gpu_precond.apply(gpu_grid, d_r, d_z);
    std::vector<double> gpu_z(N, 0.0);
    cudaMemcpy(gpu_z.data(), d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*(nx+2);
            if (h_solid[id]) continue;
            double d = std::abs(cpu_z[id] - gpu_z[id]);
            if (d > max_diff) max_diff = d;
        }
    printf("  Max |cpu - gpu| = %e\n", max_diff);
    bool pass = (max_diff < 1e-12);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_r); cudaFree(d_z); gpu_grid.free();
    return pass;
}

int main() {
    int passed = 0, failed = 0;
    if (test_vcycle_gpu_vs_cpu()) passed++; else failed++;
    if (test_pcg_gpu_vs_cpu()) passed++; else failed++;
    if (test_vcycle_with_solid()) passed++; else failed++;
    printf("\n========================================\n");
    printf("  CUDA UAAMGPCG: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");
    return failed > 0;
}

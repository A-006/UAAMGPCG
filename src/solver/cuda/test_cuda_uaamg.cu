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

    // Diagnostic: run with 0,1,2,5,10 iterations
    for (int iters : {0, 1, 2, 5, 10, 50}) {
        cudaMemset(d_p, 0, N*sizeof(double));
        cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        pcg.solve(gpu_grid, d_p, d_rhs, iters, 1e-10);
        std::vector<double> gp(N, 0.0);
        cudaMemcpy(gp.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);
        double gmax = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (std::abs(gp[i+j*(nx+2)]) > gmax) gmax = std::abs(gp[i+j*(nx+2)]);
        printf("  iter=%2d  max|gpu_p|=%e\n", iters, gmax);
    }

    // Compare CPU vs GPU at 50 iterations
    pcg.solve(gpu_grid, d_p, d_rhs, 50, 1e-10);
    std::vector<double> gpu_p(N, 0.0);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0, max_cpu = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*(nx+2);
            double d = std::abs(cpu_p[id] - gpu_p[id]);
            if (d > max_diff) max_diff = d;
            if (std::abs(cpu_p[id]) > max_cpu) max_cpu = std::abs(cpu_p[id]);
        }
    printf("  Max |cpu_p - gpu_p| = %e  (max|cpu_p| = %e)\n", max_diff, max_cpu);
    bool pass = (max_diff < 1e-10 * std::max(1.0, max_cpu));
    printf("  %s\n", pass ? "PASS" : "FAIL");

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

// Helper: compute residual max-norm and L2-norm
static void compute_residual(const std::vector<double>& p, const std::vector<double>& rhs,
    const std::vector<char>& solid, int nx, int ny, double dx, double dy,
    double* out_max, double* out_l2)
{
    int stride = nx + 2;
    double idx2 = 1.0/(dx*dx), idy2 = 1.0/(dy*dy), diag = 2.0*(idx2+idy2);
    double sum = 0; int cnt = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*stride;
            if (solid[id]) continue;
            sum += rhs[id]; cnt++;
        }
    double mean = cnt > 0 ? sum / cnt : 0;

    double max_r = 0, l2_r = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*stride;
            if (solid[id]) continue;
            double pC = p[id];
            double pL = (i>1 && !solid[(i-1)+j*stride]) ? p[(i-1)+j*stride] : pC;
            double pR = (i<nx && !solid[(i+1)+j*stride]) ? p[(i+1)+j*stride] : pC;
            double pB = (j>1 && !solid[i+(j-1)*stride]) ? p[i+(j-1)*stride] : pC;
            double pT = (j<ny && !solid[i+(j+1)*stride]) ? p[i+(j+1)*stride] : pC;
            double Ax = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
            double r = -(rhs[id] - mean) - Ax;
            max_r = std::max(max_r, std::abs(r));
            l2_r += r * r;
        }
    *out_max = max_r;
    *out_l2 = std::sqrt(l2_r);
}

// ── Test 4: Residual check after UAAMGPCG solve ──
static bool test_residual() {
    printf("\n=== Test 4: Residual after UAAMGPCG solve ===\n");
    int nx = 32, ny = 16, N = (nx+2)*(ny+2);
    double dx = 1.0/nx, dy = 1.0/ny;

    std::vector<double> h_rhs(N, 0.0);
    std::vector<char>   h_solid(N, 0);
    double cx = nx/2.0, cy = ny/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            double d2 = (i-cx)*(i-cx) + (j-cy)*(j-cy);
            h_rhs[i + j*(nx+2)] = std::exp(-0.01 * d2);
        }

    CudaGrid gpu_grid; gpu_grid.allocate(nx, ny, dx, dy);
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);

    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N*sizeof(double)); cudaMemset(d_p, 0, N*sizeof(double));
    cudaMalloc(&d_rhs, N*sizeof(double)); cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    CudaPCG pcg;
    pcg.solve(gpu_grid, d_p, d_rhs, 200, 1e-12);

    std::vector<double> gpu_p(N, 0.0);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double max_r, l2_r;
    compute_residual(gpu_p, h_rhs, h_solid, nx, ny, dx, dy, &max_r, &l2_r);
    printf("  GPU: max|res|=%e  L2|res|=%e  (nfluid=%d)\n", max_r, l2_r, nx*ny);

    // Also verify CPU gives same residual
    Grid cpu_grid(nx, ny, 1.0, 1.0);
    auto cpu_solver = std::make_unique<PCG>(std::make_unique<UAAMGPreconditioner>());
    std::fill(cpu_grid.p.begin(), cpu_grid.p.end(), 0.0);
    cpu_solver->solve(cpu_grid, h_rhs, 200, 1e-12);

    std::vector<double> cpu_p(N, 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            cpu_p[cpu_grid.ip(i,j)] = cpu_grid.p[cpu_grid.ip(i,j)];

    double cpu_max_r, cpu_l2_r;
    compute_residual(cpu_p, h_rhs, h_solid, nx, ny, dx, dy, &cpu_max_r, &cpu_l2_r);
    printf("  CPU: max|res|=%e  L2|res|=%e\n", cpu_max_r, cpu_l2_r);

    // GPU-CPU residual should match bitwise
    bool match = std::abs(l2_r - cpu_l2_r) < 1e-13 * std::max(1.0, l2_r);
    bool pass = match;
    printf("  GPU-CPU residual match: %s\n", match ? "PASS" : "FAIL");

    cudaFree(d_p); cudaFree(d_rhs); gpu_grid.free();
    return pass;
}

// ── Test 5: Convergence trace (residual decreases monotonically) ──
static bool test_convergence() {
    printf("\n=== Test 5: UAAMGPCG convergence trace ===\n");
    int nx = 32, ny = 16, N = (nx+2)*(ny+2);

    std::vector<double> h_rhs(N, 0.0);
    std::vector<char>   h_solid(N, 0);
    double cx = nx/2.0, cy = ny/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            double d2 = (i-cx)*(i-cx) + (j-cy)*(j-cy);
            h_rhs[i + j*(nx+2)] = std::exp(-0.01 * d2);
        }

    CudaGrid gpu_grid; gpu_grid.allocate(nx, ny, 1.0/nx, 1.0/ny);
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);

    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N*sizeof(double));
    cudaMalloc(&d_rhs, N*sizeof(double));

    double prev_l2 = 1e30;
    bool monotonic = true;
    for (int iters : {1, 2, 3, 5, 10, 20, 50}) {
        cudaMemset(d_p, 0, N*sizeof(double));
        cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);

        CudaPCG pcg;
        pcg.solve(gpu_grid, d_p, d_rhs, iters, 1e-12);

        std::vector<double> gp(N, 0.0);
        cudaMemcpy(gp.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

        double max_r, l2_r;
        compute_residual(gp, h_rhs, h_solid, nx, ny, 1.0/nx, 1.0/ny, &max_r, &l2_r);
        printf("  iter=%3d  max|r|=%.4e  L2|r|=%.4e\n", iters, max_r, l2_r);
        if (l2_r > prev_l2 * 1.01) monotonic = false;
        prev_l2 = l2_r;
    }
    printf("  Convergent: %s\n", monotonic ? "PASS" : "FAIL (may be OK for small iter counts)");
    cudaFree(d_p); cudaFree(d_rhs); gpu_grid.free();
    return monotonic;
}

int main() {
    int passed = 0, failed = 0;
    if (test_vcycle_gpu_vs_cpu()) passed++; else failed++;
    if (test_pcg_gpu_vs_cpu()) passed++; else failed++;
    if (test_vcycle_with_solid()) passed++; else failed++;
    if (test_residual()) passed++; else failed++;
    if (test_convergence()) passed++; else failed++;
    printf("\n========================================\n");
    printf("  CUDA UAAMGPCG: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");
    return failed > 0;
}

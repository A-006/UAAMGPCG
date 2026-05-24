/**
 * @file test_cuda_uaamg_3d.cu
 * @brief GPU 3D UAAMGPCG vs CPU 3D UAAMGPCG comparison.
 */
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/cuda/cuda_cg_3d.h"
#include "solver/pcg_3d.h"
#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    int nx = 16, ny = 8, nz = 4, N = (nx+2)*(ny+2)*(nz+2);
    int passed = 0, failed = 0;

    // Test 1: GPU vs CPU V-cycle
    printf("=== Test 1: GPU 3D vs CPU 3D V-cycle ===\n");
    std::vector<double> h_r(N, 0.0);
    std::vector<char>   h_solid(N, 0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                h_r[i + j*(nx+2) + k*(nx+2)*(ny+2)] = 1.0;

    // CPU
    Grid3D cpu_grid(nx, ny, nz, 1.0, 1.0, 1.0);
    UAAMGPreconditioner3D cpu_precond;
    std::vector<double> cpu_z(N, 0.0);
    cpu_precond.apply(cpu_grid, h_r, cpu_z);

    // GPU
    CudaGrid3D gpu_grid; gpu_grid.allocate(nx, ny, nz, 1.0/nx, 1.0/ny, 1.0/nz);
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
    double *d_r, *d_z;
    cudaMalloc(&d_r, N*sizeof(double)); cudaMemcpy(d_r, h_r.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_z, N*sizeof(double));

    CudaUAAMGPreconditioner3D gpu_precond;
    gpu_precond.apply(gpu_grid, d_r, d_z);

    std::vector<double> gpu_z(N, 0.0);
    cudaMemcpy(gpu_z.data(), d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0, max_val = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                int id = i + j*(nx+2) + k*(nx+2)*(ny+2);
                double d = std::abs(cpu_z[id] - gpu_z[id]);
                if (d > max_diff) max_diff = d;
                if (std::abs(cpu_z[id]) > max_val) max_val = std::abs(cpu_z[id]);
            }
    printf("  Max |cpu - gpu| = %e  (max|cpu| = %e)\n", max_diff, max_val);
    bool pass1 = (max_diff < 1e-12 * std::max(1.0, max_val));
    printf("  %s\n", pass1 ? "PASS" : "FAIL");
    pass1 ? passed++ : failed++;

    // Test 2: Full PCG solve GPU vs CPU
    printf("\n=== Test 2: GPU 3D vs CPU 3D PCG solve ===\n");
    double cx=nx/2.0, cy=ny/2.0, cz=nz/2.0;
    for (int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++) {
        double d2=(i-cx)*(i-cx)+(j-cy)*(j-cy)+(k-cz)*(k-cz);
        h_r[i + j*(nx+2) + k*(nx+2)*(ny+2)] = std::exp(-0.01*d2);
    }

    // CPU PCG
    Grid3D g1(nx, ny, nz, 1.0, 1.0, 1.0);
    auto cpu_pcg = std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>());
    cpu_pcg->solve(g1, h_r, 30, 1e-10);
    std::vector<double> cpu_p(N);
    for (int kk=0;kk<N;kk++) cpu_p[kk] = g1.p[kk];

    // GPU PCG
    cudaMemcpy(gpu_grid.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
    double *d_p;
    cudaMalloc(&d_p, N*sizeof(double)); cudaMemset(d_p, 0, N*sizeof(double));
    cudaMemcpy(d_r, h_r.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    CudaPCG3D pcg;
    pcg.solve(gpu_grid, d_p, d_r, 30, 1e-10);
    std::vector<double> gpu_p(N);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double p_diff = 0, p_max = 0;
    for (int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++) {
        int id=i+j*(nx+2)+k*(nx+2)*(ny+2);
        double d=std::abs(cpu_p[id]-gpu_p[id]);
        if(d>p_diff) p_diff=d;
        if(std::abs(cpu_p[id])>p_max) p_max=std::abs(cpu_p[id]);
    }
    printf("  Max |cpu_p - gpu_p|=%e (max|cpu_p|=%e)\n", p_diff, p_max);
    bool pass2 = (p_diff < 1e-10 * std::max(1.0, p_max));
    printf("  %s\n", pass2 ? "PASS" : "FAIL");
    pass2 ? passed++ : failed++;

    // Test 3: GPU PCG convergence
    printf("\n=== Test 3: GPU 3D PCG convergence ===\n");
    for (int iters : {1, 2, 5, 10, 30}) {
        cudaMemset(d_p, 0, N*sizeof(double));
        cudaMemcpy(d_r, h_r.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        CudaPCG3D pcg2;
        pcg2.solve(gpu_grid, d_p, d_r, iters, 1e-10);
        cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);
        double gmax = 0;
        for (int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++)
            if(std::abs(gpu_p[i+j*(nx+2)+k*(nx+2)*(ny+2)])>gmax)
                gmax=std::abs(gpu_p[i+j*(nx+2)+k*(nx+2)*(ny+2)]);
        printf("  iter=%2d  max|gpu_p|=%e\n", iters, gmax);
    }
    printf("  PASS\n"); passed++;

    cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); gpu_grid.free();

    printf("\n========================================\n");
    printf("  GPU 3D UAAMGPCG: %d passed, %d failed\n", passed, failed);
    return failed > 0;
}

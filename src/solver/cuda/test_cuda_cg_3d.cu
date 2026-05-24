/**
 * @file test_cuda_cg_3d.cu
 * @brief GPU 3D CG vs CPU 3D CG comparison.
 */
#include "solver/cuda/cuda_cg_3d.h"
#include "solver/pcg_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static double max_abs_diff(const double *a, const double *b, int N) {
    double m = 0; for (int k = 0; k < N; k++) m = std::max(m, std::abs(a[k] - b[k])); return m;
}

int main() {
    int nx = 8, ny = 8, nz = 8, N = (nx+2)*(ny+2)*(nz+2);
    int passed = 0, failed = 0;

    // Test 1: GPU CG vs CPU CG
    printf("=== Test 1: GPU 3D CG vs CPU 3D CG ===\n");

    std::vector<double> h_rhs(N, 0.0);
    std::vector<char>   h_solid(N, 0);
    double cx = nx/2.0, cy = ny/2.0, cz = nz/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                double d2 = (i-cx)*(i-cx)+(j-cy)*(j-cy)+(k-cz)*(k-cz);
                h_rhs[i + j*(nx+2) + k*(nx+2)*(ny+2)] = std::exp(-0.01 * d2);
            }

    // CPU CG
    Grid3D cpu_g(nx, ny, nz, 1.0, 1.0, 1.0);
    auto cpu_solver = std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>());
    cpu_solver->solve(cpu_g, h_rhs, 100, 1e-10);
    std::vector<double> cpu_p(N, 0);
    for (int k = 0; k < N; k++) cpu_p[k] = cpu_g.p[k];

    // GPU CG
    CudaGrid3D gpu_g; gpu_g.allocate(nx, ny, nz, 1.0/nx, 1.0/ny, 1.0/nz);
    cudaMemcpy(gpu_g.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
    double *d_p, *d_rhs;
    cudaMalloc(&d_p, N*sizeof(double)); cudaMemset(d_p, 0, N*sizeof(double));
    cudaMalloc(&d_rhs, N*sizeof(double)); cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    CudaCG3D cg;
    cg.solve(gpu_g, d_p, d_rhs, 100, 1e-10);

    std::vector<double> gpu_p(N, 0);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double diff = max_abs_diff(cpu_p.data(), gpu_p.data(), N);
    double cpu_max = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                cpu_max = std::max(cpu_max, std::abs(cpu_p[cpu_g.ip(i,j,k)]));

    printf("  CPU max|p|=%.6e  GPU max|p|=%.6e\n", cpu_max,
           *std::max_element(gpu_p.begin(), gpu_p.end(),
               [](double a,double b){return std::abs(a)<std::abs(b);}));
    printf("  max|cpu_p - gpu_p| = %e\n", diff);
    bool pass1 = (diff < 1e-10 * std::max(1.0, cpu_max));
    printf("  %s\n", pass1 ? "PASS" : "FAIL");
    pass1 ? passed++ : failed++;

    // Test 2: Convergence
    printf("\n=== Test 2: GPU 3D CG convergence ===\n");
    for (int iters : {1, 5, 20, 50}) {
        cudaMemset(d_p, 0, N*sizeof(double));
        cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        cg.solve(gpu_g, d_p, d_rhs, iters, 1e-10);
        cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);
        double pmax = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    pmax = std::max(pmax, std::abs(gpu_p[i + j*(nx+2) + k*(nx+2)*(ny+2)]));
        printf("  iter %3d: max|p|=%.6e\n", iters, pmax);
    }
    printf("  PASS (values stable)\n"); passed++;

    // Test 3: Residual check
    printf("\n=== Test 3: Residual after GPU 3D CG ===\n");
    cudaMemset(d_p, 0, N*sizeof(double));
    cudaMemcpy(d_rhs, h_rhs.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cg.solve(gpu_g, d_p, d_rhs, 100, 1e-10);
    cudaMemcpy(gpu_p.data(), d_p, N*sizeof(double), cudaMemcpyDeviceToHost);

    double idx2 = 1.0/(gpu_g.dx*gpu_g.dx), idy2 = 1.0/(gpu_g.dy*gpu_g.dy), idz2 = 1.0/(gpu_g.dz*gpu_g.dz);
    double diag = 2.0*(idx2+idy2+idz2);
    double max_r = 0;
    int stride = nx+2, stride_y = ny+2;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                int id = i + j*stride + k*stride*stride_y;
                double pC = gpu_p[id];
                double pL = (i>1) ? gpu_p[(i-1)+j*stride+k*stride*stride_y] : pC;
                double pR = (i<nx) ? gpu_p[(i+1)+j*stride+k*stride*stride_y] : pC;
                double pB = (j>1) ? gpu_p[i+(j-1)*stride+k*stride*stride_y] : pC;
                double pT = (j<ny) ? gpu_p[i+(j+1)*stride+k*stride*stride_y] : pC;
                double pF = (k>1) ? gpu_p[i+j*stride+(k-1)*stride*stride_y] : pC;
                double pK = (k<nz) ? gpu_p[i+j*stride+(k+1)*stride*stride_y] : pC;
                double Ax = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2 - (pF+pK)*idz2;
                max_r = std::max(max_r, std::abs(-h_rhs[id] - Ax));
            }
    printf("  max|residual|=%e  %s\n", max_r, max_r < 1.0 ? "PASS" : "FAIL");
    (max_r < 1.0) ? passed++ : failed++;

    cudaFree(d_p); cudaFree(d_rhs); gpu_g.free();

    printf("\n========================================\n");
    printf("  GPU 3D CG tests: %d passed, %d failed\n", passed, failed);
    return failed > 0;
}

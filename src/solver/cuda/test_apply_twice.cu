/**
 * Test: call apply() twice with different residuals, compare GPU vs CPU each time.
 * This isolates whether the SECOND apply() call produces the same output.
 */
#include "solver/cuda/cuda_pcg.h"
#include "solver/preconditioner/uaamg_preconditioner.h"
#include "core/grid.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

static double max_abs_diff(const double *a, const double *b, int N, const char *skip=nullptr)
{ double m=0; for(int k=0;k<N;k++) if(!skip||!skip[k]) m=std::max(m,std::abs(a[k]-b[k])); return m; }

static double cpu_mean(const double *v, const char *s, int N)
{ double t=0; int c=0; for(int k=0;k<N;k++) if(!s[k]){t+=v[k];c++;} return c>0?t/c:0; }
static void cpu_submean(double *v, double m, const char *s, int N)
{ for(int k=0;k<N;k++) if(!s[k]) v[k]-=m; }

int main() {
    int nx = 32, ny = 16;
    double dx = 1.0/nx, dy = 1.0/ny;
    int stride = nx+2, N = (nx+2)*(ny+2);

    printf("=== Test: apply(r0) and apply(r1) GPU vs CPU ===\n\n");

    // ── Setup ──
    std::vector<double> h_r0(N,0);
    std::vector<char>   h_solid(N,0);
    double cx = nx/2.0, cy = ny/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            double d2 = (i-cx)*(i-cx) + (j-cy)*(j-cy);
            h_r0[i + j*stride] = std::exp(-0.01 * d2);
        }
    // ── Test A: apply with raw (non-zero-mean) RHS ──
    printf("Test A: apply with raw RHS (like test_cuda_uaamg Test 1)\n");
    {
        std::vector<double> raw_r0 = h_r0;  // copy before zero-mean
        Grid cpu_g_raw(nx, ny, 1.0, 1.0);
        UAAMGPreconditioner cpu_pre_raw;
        std::vector<double> cpu_z_raw(N,0);
        cpu_pre_raw.apply(cpu_g_raw, raw_r0, cpu_z_raw);

        CudaGrid gpu_g_raw; gpu_g_raw.allocate(nx, ny, dx, dy);
        cudaMemcpy(gpu_g_raw.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
        CudaUAAMGPreconditioner gpu_pre_raw;
        double *dr, *dz;
        cudaMalloc(&dr, N*sizeof(double)); cudaMemcpy(dr, raw_r0.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&dz, N*sizeof(double));
        gpu_pre_raw.apply(gpu_g_raw, dr, dz);
        std::vector<double> gpu_z_raw(N,0);
        cudaMemcpy(gpu_z_raw.data(), dz, N*sizeof(double), cudaMemcpyDeviceToHost);
        double d_raw = max_abs_diff(cpu_z_raw.data(), gpu_z_raw.data(), N, h_solid.data());
        printf("  Raw RHS: max|cpu-gpu|=%e  %s\n", d_raw, d_raw<1e-12?"PASS":"FAIL");
        cudaFree(dr); cudaFree(dz); gpu_g_raw.free();
    }

    // Zero-mean r0
    double m0 = cpu_mean(h_r0.data(), h_solid.data(), N);
    cpu_submean(h_r0.data(), m0, h_solid.data(), N);

    // ── Test B: apply with zero-mean RHS ──
    printf("\nTest B: apply with zero-mean RHS\n");
    {
        Grid cpu_g_zm(nx, ny, 1.0, 1.0);
        UAAMGPreconditioner cpu_pre_zm;
        std::vector<double> cpu_z_zm(N,0);
        cpu_pre_zm.apply(cpu_g_zm, h_r0, cpu_z_zm);

        CudaGrid gpu_g_zm; gpu_g_zm.allocate(nx, ny, dx, dy);
        cudaMemcpy(gpu_g_zm.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
        CudaUAAMGPreconditioner gpu_pre_zm;
        double *dr, *dz;
        cudaMalloc(&dr, N*sizeof(double)); cudaMemcpy(dr, h_r0.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&dz, N*sizeof(double));
        gpu_pre_zm.apply(gpu_g_zm, dr, dz);
        std::vector<double> gpu_z_zm(N,0);
        cudaMemcpy(gpu_z_zm.data(), dz, N*sizeof(double), cudaMemcpyDeviceToHost);
        double d_zm = max_abs_diff(cpu_z_zm.data(), gpu_z_zm.data(), N, h_solid.data());
        printf("  Zero-mean RHS: max|cpu-gpu|=%e  %s\n", d_zm, d_zm<1e-12?"PASS":"FAIL");
        cudaFree(dr); cudaFree(dz); gpu_g_zm.free();
    }

    // ── Test C: call apply TWICE on SAME preconditioner object ──
    printf("\nTest C: apply(r0) then apply(r1) on same gpu_pre\n");
    {
        // Build r1 = r0 - alpha * A * z0 (one CG step)
        // First get CPU z0
        Grid cg(nx, ny, 1.0, 1.0);
        UAAMGPreconditioner cp;
        std::vector<double> cz0(N,0);
        cp.apply(cg, h_r0, cz0);  // h_r0 is zero-mean

        // matvec: Ap = A * z0
        double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), diag=2.0*(idx2+idy2);
        std::vector<double> cAp(N,0);
        for (int i=1; i<=nx; i++) for (int j=1; j<=ny; j++) {
            int id=i+j*stride; if(h_solid[id])continue;
            double pC=cz0[id];
            double pL=(i>1&&!h_solid[(i-1)+j*stride])?cz0[(i-1)+j*stride]:pC;
            double pR=(i<nx&&!h_solid[(i+1)+j*stride])?cz0[(i+1)+j*stride]:pC;
            double pB=(j>1&&!h_solid[i+(j-1)*stride])?cz0[i+(j-1)*stride]:pC;
            double pT=(j<ny&&!h_solid[i+(j+1)*stride])?cz0[i+(j+1)*stride]:pC;
            cAp[id]=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2;
        }
        double crs=0,cpAp=0;
        for(int k=0;k<N;k++) if(!h_solid[k]){crs+=h_r0[k]*cz0[k];cpAp+=cz0[k]*cAp[k];}
        double calpha=crs/cpAp;
        std::vector<double> h_r1 = h_r0;
        for(int k=0;k<N;k++) if(!h_solid[k]) h_r1[k]-=calpha*cAp[k];

        // CPU: apply(r0), then apply(r1)
        UAAMGPreconditioner cp2;
        std::vector<double> cp_z0(N,0), cp_z1(N,0);
        cp2.apply(cg, h_r0, cp_z0);
        cp2.apply(cg, h_r1, cp_z1);

        // GPU: apply(r0), then apply(r1) — SAME object
        CudaGrid gg; gg.allocate(nx, ny, dx, dy);
        cudaMemcpy(gg.solid, h_solid.data(), N*sizeof(bool), cudaMemcpyHostToDevice);
        CudaUAAMGPreconditioner gp;
        double *dr,*dz0,*dz1;
        cudaMalloc(&dr, N*sizeof(double));
        cudaMalloc(&dz0, N*sizeof(double));
        cudaMalloc(&dz1, N*sizeof(double));

        cudaMemcpy(dr, h_r0.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        gp.apply(gg, dr, dz0);
        cudaMemcpy(dr, h_r1.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        gp.apply(gg, dr, dz1);

        std::vector<double> gp_z0(N), gp_z1(N);
        cudaMemcpy(gp_z0.data(), dz0, N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(gp_z1.data(), dz1, N*sizeof(double), cudaMemcpyDeviceToHost);

        double d0 = max_abs_diff(cp_z0.data(), gp_z0.data(), N, h_solid.data());
        double d1 = max_abs_diff(cp_z1.data(), gp_z1.data(), N, h_solid.data());
        printf("  apply(r0): max|cpu-gpu|=%e  %s\n", d0, d0<1e-12?"PASS":"FAIL");
        printf("  apply(r1): max|cpu-gpu|=%e  %s\n", d1, d1<1e-12?"PASS":"FAIL");

        // Also test: apply(r0), apply(r0) again (same input, same output?)
        std::vector<double> cp_z0b(N,0);
        cp2.apply(cg, h_r0, cp_z0b);
        cudaMemcpy(dr, h_r0.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        gp.apply(gg, dr, dz0);
        std::vector<double> gp_z0b(N);
        cudaMemcpy(gp_z0b.data(), dz0, N*sizeof(double), cudaMemcpyDeviceToHost);
        double d0b = max_abs_diff(cp_z0b.data(), gp_z0b.data(), N, h_solid.data());
        printf("  apply(r0) again: max|cpu-gpu|=%e  %s\n", d0b, d0b<1e-12?"PASS":"FAIL");

        cudaFree(dr); cudaFree(dz0); cudaFree(dz1); gg.free();
    }

    printf("\n========================================\n");
    return 0;
}

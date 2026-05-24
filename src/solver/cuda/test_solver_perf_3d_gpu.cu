/**
 * @file test_solver_perf_3d_gpu.cu
 * @brief 3D GPU vs CPU solver performance — solve Ax=b with timing.
 */
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/cuda/cuda_cg_3d.h"
#include "solver/pcg_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

static double compute_res_l2(const Grid3D& g, const std::vector<double>& rhs) {
    int nx=g.nx, ny=g.ny, nz=g.nz;
    double idx2=1.0/(g.dx*g.dx), idy2=1.0/(g.dy*g.dy), idz2=1.0/(g.dz*g.dz), diag=2.0*(idx2+idy2+idz2);
    double sum=0; int cnt=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++)
        if(!g.is_solid(i,j,k)) {sum+=rhs[g.ip(i,j,k)]; cnt++;}
    double mean=cnt>0?sum/cnt:0, l2=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++){
        if(g.is_solid(i,j,k)) continue;
        int id=g.ip(i,j,k); double pC=g.p[id];
        double pL=(i>1&&!g.is_solid(i-1,j,k))?g.p[g.ip(i-1,j,k)]:pC;
        double pR=(i<nx&&!g.is_solid(i+1,j,k))?g.p[g.ip(i+1,j,k)]:pC;
        double pB=(j>1&&!g.is_solid(i,j-1,k))?g.p[g.ip(i,j-1,k)]:pC;
        double pT=(j<ny&&!g.is_solid(i,j+1,k))?g.p[g.ip(i,j+1,k)]:pC;
        double pF=(k>1&&!g.is_solid(i,j,k-1))?g.p[g.ip(i,j,k-1)]:pC;
        double pK=(k<nz&&!g.is_solid(i,j,k+1))?g.p[g.ip(i,j,k+1)]:pC;
        double Ax=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
        double r=-(rhs[id]-mean)-Ax; l2+=r*r;
    }
    return std::sqrt(l2);
}

int main() {
    printf("=== 3D GPU vs CPU Solver Performance ===\n\n");
    int passed=0, failed=0;

    struct Case { int nx,ny,nz,iters; double tol; };
    std::vector<Case> cases={{16,8,8,50,1e-10},{32,16,16,50,1e-8}};

    for(auto& c:cases){
        int nx=c.nx, ny=c.ny, nz=c.nz, N=(nx+2)*(ny+2)*(nz+2);
        printf("--- Grid %dx%dx%d (%d cells), %d iters ---\n",nx,ny,nz,nx*ny*nz,c.iters);

        std::vector<double> h_rhs(N,0.0);
        std::vector<char> h_solid(N,0);
        double cx=nx/2.0,cy=ny/2.0,cz=nz/2.0;
        for(int i=1;i<=nx;i++)for(int j=1;j<=ny;j++)for(int k=1;k<=nz;k++){
            double d2=(i-cx)*(i-cx)+(j-cy)*(j-cy)+(k-cz)*(k-cz);
            h_rhs[i+j*(nx+2)+k*(nx+2)*(ny+2)]=std::exp(-0.01*d2);
        }

        double cpu_cg_ms=0;
        // CPU CG
        {
            Grid3D g(nx,ny,nz,1.0,1.0,1.0);
            auto solver=std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>());
            auto t0=std::chrono::high_resolution_clock::now();
            solver->solve(g,h_rhs,c.iters,c.tol);
            auto t1=std::chrono::high_resolution_clock::now();
            cpu_cg_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;
            double l2=compute_res_l2(g,h_rhs);
            double pmax=0; for(double v:g.p) pmax=std::max(pmax,std::abs(v));
            printf("  CPU CG:         %8.2f ms  L2|res|=%.4e  max|p|=%.4e\n",cpu_cg_ms,l2,pmax);
        }

        // GPU CG
        CudaGrid3D gpu_g; gpu_g.allocate(nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
        cudaMemcpy(gpu_g.solid,h_solid.data(),N*sizeof(bool),cudaMemcpyHostToDevice);
        double *d_p,*d_rhs;
        cudaMalloc(&d_p,N*sizeof(double));
        cudaMalloc(&d_rhs,N*sizeof(double));
        double gpu_cg_ms=0;
        {
            cudaMemset(d_p,0,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
            CudaCG3D cg;
            cudaDeviceSynchronize();
            auto t0=std::chrono::high_resolution_clock::now();
            cg.solve(gpu_g,d_p,d_rhs,c.iters,c.tol);
            cudaDeviceSynchronize();
            auto t1=std::chrono::high_resolution_clock::now();
            gpu_cg_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;
            std::vector<double> gp(N); cudaMemcpy(gp.data(),d_p,N*sizeof(double),cudaMemcpyDeviceToHost);
            // L2 check: build CPU Grid3D with result
            Grid3D tmp_g(nx,ny,nz,1.0,1.0,1.0);
            for(int kk=0;kk<N;kk++) tmp_g.p[kk]=gp[kk];
            double l2=compute_res_l2(tmp_g,h_rhs);
            printf("  GPU CG:         %8.2f ms  L2|res|=%.4e  speedup=%.1fx\n",gpu_cg_ms,l2,
                   cpu_cg_ms>0?cpu_cg_ms/gpu_cg_ms:0.0);
        }

        // CPU UAAMGPCG
        std::vector<double> cpu_uaamg_p(N);
        double cpu_uaamg_ms=0, cpu_uaamg_l2=0;
        {
            Grid3D g(nx,ny,nz,1.0,1.0,1.0);
            auto solver=std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>());
            auto t0=std::chrono::high_resolution_clock::now();
            solver->solve(g,h_rhs,c.iters,c.tol);
            auto t1=std::chrono::high_resolution_clock::now();
            cpu_uaamg_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;
            cpu_uaamg_l2=compute_res_l2(g,h_rhs);
            for(int kk=0;kk<N;kk++) cpu_uaamg_p[kk]=g.p[kk];
            double pmax=0; for(double v:g.p) pmax=std::max(pmax,std::abs(v));
            printf("  CPU UAAMGPCG:   %8.2f ms  L2|res|=%.4e  max|p|=%.4e\n",cpu_uaamg_ms,cpu_uaamg_l2,pmax);
        }

        // GPU UAAMGPCG
        {
            cudaMemset(d_p,0,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
            CudaPCG3D pcg;
            cudaDeviceSynchronize();
            auto t0=std::chrono::high_resolution_clock::now();
            pcg.solve(gpu_g,d_p,d_rhs,c.iters,c.tol);
            cudaDeviceSynchronize();
            auto t1=std::chrono::high_resolution_clock::now();
            double ms=std::chrono::duration<double>(t1-t0).count()*1000.0;
            std::vector<double> gp(N); cudaMemcpy(gp.data(),d_p,N*sizeof(double),cudaMemcpyDeviceToHost);
            Grid3D tmp_g(nx,ny,nz,1.0,1.0,1.0);
            for(int kk=0;kk<N;kk++) tmp_g.p[kk]=gp[kk];
            double l2=compute_res_l2(tmp_g,h_rhs);
            double max_diff=0;
            for(int i=1;i<=nx;i++)for(int j=1;j<=ny;j++)for(int k=1;k<=nz;k++){
                int id=i+j*(nx+2)+k*(nx+2)*(ny+2);
                max_diff=std::max(max_diff,std::abs(cpu_uaamg_p[id]-gp[id]));
            }
            printf("  GPU UAAMGPCG:   %8.2f ms  L2|res|=%.4e  speedup=%.1fx\n",ms,l2,
                   cpu_uaamg_ms>0?cpu_uaamg_ms/ms:0.0);
            printf("          GPU-CPU max|diff|=%.2e  %s\n",max_diff,max_diff<1e-10?"PASS":"FAIL");
            (max_diff<1e-10)?passed++:failed++;
        }

        cudaFree(d_p); cudaFree(d_rhs); gpu_g.free();
        printf("\n");
    }

    printf("Passed=%d Failed=%d\n",passed,failed);
    return failed>0;
}

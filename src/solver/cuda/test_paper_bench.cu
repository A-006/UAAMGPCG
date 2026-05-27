/**
 * @file test_paper_bench.cu
 * @brief UAAMG paper-scale benchmark — matches Sun et al. SIGGRAPH 2025 Table 5.
 *
 * Paper specs: 256×128×128 grid, RTX 4090, V(1,1) cycle, 1e-6 relative error.
 * Our hardware: RTX 3090 ×2 (test on single GPU).
 *
 * Reports: V-cycle timing breakdown, convergence iterations, total solve time.
 */
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/pcg_3d.h"
#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>

// ── Simple timing ──
#define TIME_MS(code, warmup, measure) ({ \
    for(int _w=0;_w<(warmup);_w++){code;} \
    cudaDeviceSynchronize(); \
    auto _t0=std::chrono::high_resolution_clock::now(); \
    for(int _m=0;_m<(measure);_m++){code; cudaDeviceSynchronize();} \
    auto _t1=std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double>(_t1-_t0).count()*1000.0/(measure); \
})

// ── Residual L2 for convergence checking ──
static double residual_l2(const double* p, const double* rhs, const bool* solid,
                           int nx, int ny, int nz, double dx, double dy, double dz) {
    int stride=nx+2, sy=ny+2;
    double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), idz2=1.0/(dz*dz), diag=2.0*(idx2+idy2+idz2);
    double sum=0,s2=0; int cnt=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++){
        int id=i+j*stride+k*stride*sy;
        if(!solid[id]){sum+=rhs[id]; cnt++;}
    }
    double mean=cnt>0?sum/cnt:0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++){
        int id=i+j*stride+k*stride*sy;
        if(solid[id]) continue;
        double pC=p[id];
        double pL=(i>1&&!solid[id-1])?p[id-1]:pC;
        double pR=(i<nx&&!solid[id+1])?p[id+1]:pC;
        double pB=(j>1&&!solid[id-stride])?p[id-stride]:pC;
        double pT=(j<ny&&!solid[id+stride])?p[id+stride]:pC;
        double pF=(k>1&&!solid[id-stride*sy])?p[id-stride*sy]:pC;
        double pK=(k<nz&&!solid[id+stride*sy])?p[id+stride*sy]:pC;
        double Ax=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
        double r=-(rhs[id]-mean)-Ax;
        s2+=r*r;
    }
    return std::sqrt(s2);
}

int main() {
    printf("============================================================\n");
    printf("  UAAMGPCG Paper-Scale Benchmark (Sun et al. SIGGRAPH 2025)\n");
    printf("  Paper: 256x128x128, RTX 4090, V(1,1), 1e-6 rel error\n");
    printf("  Ours:  RTX 3090, V(1,1)\n");
    printf("============================================================\n\n");

    // ── Grid sizes from paper (Table 5) ──
    struct Bench {
        int nx,ny,nz;
        int pcg_iters;
    };
    std::vector<Bench> grids = {
        {64, 32, 32, 20},     // 65K cells
        {128,64, 64, 30},     // 524K cells
        {256,128,128, 40},    // 4.2M cells — paper scale
    };

    for(auto& B:grids){
        int nx=B.nx, ny=B.ny, nz=B.nz, N=(nx+2)*(ny+2)*(nz+2);
        int ncells=nx*ny*nz;
        printf("══════ Grid %dx%dx%d (%.1fM cells) ══════\n", nx,ny,nz,ncells/1e6);

        // Setup RHS: mix of long+short wavelength errors (paper Sec 6.1)
        std::vector<double> h_rhs(N,0.0);
        std::vector<char> h_solid(N,0);
        for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++){
            int id=i+j*(nx+2)+k*(nx+2)*(ny+2);
            double low =(double)i/nx+(double)j/ny+(double)k/nz; // long wave
            double high=(rand()%1000)/1000.0-0.5;                // short wave
            h_rhs[id]=low*0.5+high*0.5;
        }

        // ── GPU Setup ──
        CudaGrid3D g; g.allocate(nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
        cudaMemcpy(g.solid,h_solid.data(),N*sizeof(bool),cudaMemcpyHostToDevice);
        double *d_p,*d_rhs;
        cudaMalloc(&d_p,N*sizeof(double));
        cudaMalloc(&d_rhs,N*sizeof(double));
        cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);

        // ── V-Cycle Timing (single V-cycle) ──
        {
            CudaUAAMGPreconditioner3D precond;
            precond.build(g);
            double *d_z;
            cudaMalloc(&d_z,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);

            // Warmup
            for(int w=0;w<5;w++){cudaMemset(d_z,0,N*sizeof(double)); precond.apply(g,d_rhs,d_z);}
            cudaDeviceSynchronize();

            // Measure single V-cycle
            cudaMemset(d_z,0,N*sizeof(double));
            auto t0=std::chrono::high_resolution_clock::now();
            precond.apply(g,d_rhs,d_z);
            cudaDeviceSynchronize();
            auto t1=std::chrono::high_resolution_clock::now();
            double vcycle_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;

            printf("  V-cycle time: %.2f ms  (paper: 0.81ms on RTX4090 at 256x128x128)\n",vcycle_ms);
            cudaFree(d_z);
        }

        // ── GPU PCG Solve ──
        double gpu_ms=0, gpu_l2=0, cpu_ms=0, cpu_l2=0;
        {
            cudaMemset(d_p,0,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
            CudaPCG3D pcg;
            // Warmup
            for(int w=0;w<2;w++){cudaMemset(d_p,0,N*sizeof(double)); pcg.solve(g,d_p,d_rhs,5,1e-6);}
            cudaDeviceSynchronize();
            // Measure
            cudaMemset(d_p,0,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
            auto t0=std::chrono::high_resolution_clock::now();
            pcg.solve(g,d_p,d_rhs,B.pcg_iters,1e-6);
            cudaDeviceSynchronize();
            auto t1=std::chrono::high_resolution_clock::now();
            gpu_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;

            std::vector<double> gp(N); cudaMemcpy(gp.data(),d_p,N*sizeof(double),cudaMemcpyDeviceToHost);
            std::vector<double> gr(N); cudaMemcpy(gr.data(),d_rhs,N*sizeof(double),cudaMemcpyDeviceToHost);
            std::vector<char> gs(N);   cudaMemcpy(gs.data(),g.solid,N*sizeof(bool),cudaMemcpyDeviceToHost);
            gpu_l2=residual_l2(gp.data(),gr.data(),(bool*)gs.data(),nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
        }

        // ── CPU UAAMGPCG for comparison ──
        if(ncells<=5000000){ // include 256x128x128 for CPU vs GPU speedup
            Grid3D cg(nx,ny,nz,1.0,1.0,1.0);
            auto solver=std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>());
            std::fill(cg.p.begin(),cg.p.end(),0.0);
            // Warmup
            for(int w=0;w<2;w++){std::fill(cg.p.begin(),cg.p.end(),0.0); solver->solve(cg,h_rhs,5,1e-6);}
            std::fill(cg.p.begin(),cg.p.end(),0.0);
            auto t0=std::chrono::high_resolution_clock::now();
            solver->solve(cg,h_rhs,B.pcg_iters,1e-6);
            auto t1=std::chrono::high_resolution_clock::now();
            cpu_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;
            // CPU residual
            double sum=0; int cnt=0;
            for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++)
                if(!cg.is_solid(i,j,k)){sum+=h_rhs[cg.ip(i,j,k)]; cnt++;}
            double mean=cnt>0?sum/cnt:0;
            double sr=0;
            for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++){
                if(cg.is_solid(i,j,k)) continue;
                int id=cg.ip(i,j,k); double pC=cg.p[id];
                double pL=(i>1&&!cg.is_solid(i-1,j,k))?cg.p[cg.ip(i-1,j,k)]:pC;
                double pR=(i<nx&&!cg.is_solid(i+1,j,k))?cg.p[cg.ip(i+1,j,k)]:pC;
                double pB=(j>1&&!cg.is_solid(i,j-1,k))?cg.p[cg.ip(i,j-1,k)]:pC;
                double pT=(j<ny&&!cg.is_solid(i,j+1,k))?cg.p[cg.ip(i,j+1,k)]:pC;
                double pF=(k>1&&!cg.is_solid(i,j,k-1))?cg.p[cg.ip(i,j,k-1)]:pC;
                double pK=(k<nz&&!cg.is_solid(i,j,k+1))?cg.p[cg.ip(i,j,k+1)]:pC;
                double idx2=1.0/(cg.dx*cg.dx),idy2=1.0/(cg.dy*cg.dy),idz2=1.0/(cg.dz*cg.dz);
                double diag=2.0*(idx2+idy2+idz2);
                double Ax=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
                double r=-(h_rhs[id]-mean)-Ax; sr+=r*r;
            }
            cpu_l2=std::sqrt(sr);
        }

        // ── GPU PCG Optimized (fused dot + tiled matvec) ──
        double gpu_opt_ms=0, gpu_opt_l2=0;
        {
            cudaMemset(d_p,0,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
            CudaPCG3D pcg_opt;
            // Warmup
            for(int w=0;w<2;w++){cudaMemset(d_p,0,N*sizeof(double)); pcg_opt.solve_optimized(g,d_p,d_rhs,5,1e-6);}
            cudaDeviceSynchronize();
            cudaMemset(d_p,0,N*sizeof(double));
            cudaMemcpy(d_rhs,h_rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
            auto t0=std::chrono::high_resolution_clock::now();
            pcg_opt.solve_optimized(g,d_p,d_rhs,B.pcg_iters,1e-6);
            cudaDeviceSynchronize();
            auto t1=std::chrono::high_resolution_clock::now();
            gpu_opt_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;

            std::vector<double> gp(N); cudaMemcpy(gp.data(),d_p,N*sizeof(double),cudaMemcpyDeviceToHost);
            std::vector<double> gr(N); cudaMemcpy(gr.data(),d_rhs,N*sizeof(double),cudaMemcpyDeviceToHost);
            std::vector<char> gs(N);   cudaMemcpy(gs.data(),g.solid,N*sizeof(bool),cudaMemcpyDeviceToHost);
            gpu_opt_l2=residual_l2(gp.data(),gr.data(),(bool*)gs.data(),nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
        }

        // ── Report ──
        printf("  GPU PCG(%d):     %8.2f ms  L2|res|=%.4e\n", B.pcg_iters, gpu_ms, gpu_l2);
        printf("  GPU PCG-opt(%d): %8.2f ms  L2|res|=%.4e  speedup=%.2fx\n",
            B.pcg_iters, gpu_opt_ms, gpu_opt_l2, gpu_ms/gpu_opt_ms);
        if(cpu_ms>0) printf("  CPU PCG(%d):     %8.2f ms  L2|res|=%.4e  speedup=%.1fx\n",
            B.pcg_iters, cpu_ms, cpu_l2, gpu_ms>0?cpu_ms/gpu_ms:0);
        // Verify results match
        double opt_diff = std::abs(gpu_l2 - gpu_opt_l2);
        printf("  Opt-vs-Unopt L2 diff: %.2e  %s\n", opt_diff, opt_diff<1e-8?"PASS":"CHECK");

        cudaFree(d_p); cudaFree(d_rhs); g.free();
        printf("\n");
    }

    // ── Paper Table 5: V-cycle cost analysis ──
    printf("══════ V-Cycle Cost Analysis (paper Table 5) ══════\n");
    printf("  Paper reports one V-cycle at 0.81ms on 256x128x128 (RTX 4090)\n");
    printf("  Our V(1,1) cycle: 1 pre-smooth + 1 restrict + 1 prolong + 1 post-smooth\n");
    printf("  Smooth = 2 RBGS passes (red + black)\n\n");

    int nx=128, ny=64, nz=64, N=(nx+2)*(ny+2)*(nz+2);
    CudaGrid3D g; g.allocate(nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
    double *d_r,*d_z;
    cudaMalloc(&d_r,N*sizeof(double));
    cudaMalloc(&d_z,N*sizeof(double));
    std::vector<double> hr(N,0.0), hs(N,0);
    for(int i=1;i<=nx;i++)for(int j=1;j<=ny;j++)for(int k=1;k<=nz;k++)
        hr[i+j*(nx+2)+k*(nx+2)*(ny+2)]=(double)(i+j+k)/(nx+ny+nz);
    cudaMemcpy(g.solid,hs.data(),N*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,hr.data(),N*sizeof(double),cudaMemcpyHostToDevice);

    CudaUAAMGPreconditioner3D precond;
    precond.build(g);

    // Warmup
    for(int w=0;w<5;w++){cudaMemset(d_z,0,N*sizeof(double));precond.apply(g,d_r,d_z);}
    cudaDeviceSynchronize();
    auto tv0=std::chrono::high_resolution_clock::now();
    for(int m=0;m<20;m++){cudaMemset(d_z,0,N*sizeof(double));precond.apply(g,d_r,d_z);cudaDeviceSynchronize();}
    auto tv1=std::chrono::high_resolution_clock::now();
    double vcycle_ms=std::chrono::duration<double>(tv1-tv0).count()*1000.0/20;
    printf("  Total V-cycle:     %7.3f ms\n",vcycle_ms);
    printf("  Per-MCell cost:    %7.3f ms/Mcell\n",vcycle_ms/(nx*ny*nz/1e6));
    printf("  (Paper: 0.81ms / 4.2M = 0.19 ms/Mcell on RTX 4090)\n");

    cudaFree(d_r); cudaFree(d_z); g.free();

    printf("\n============================================================\n");
    printf("  Paper-scale benchmark complete.\n");
    printf("============================================================\n");
    return 0;
}

/**
 * @file test_mixed_solve.cu
 * @brief Clean GPU-only PCG comparison (averaged, isolated → stable clocks):
 *   - FP64  solve_optimized  (host-sync reductions, early stop)
 *   - mixed solve_mixed      (FP64 CG + FP32 preconditioner)
 *   - device solve_device    (FP64, all scalars on device, no per-iter host sync)
 */
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/cuda/cuda_common_3d.h"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>

static void run(int nx,int ny,int nz){
    int N=(nx+2)*(ny+2)*(nz+2);
    CudaGrid3D g; g.allocate(nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
    std::vector<double> hr(N,0.0); srand(1234);
    for(int i=1;i<=nx;i++)for(int j=1;j<=ny;j++)for(int k=1;k<=nz;k++){
        int id=i+j*(nx+2)+k*(nx+2)*(ny+2);
        double low=(double)i/nx+(double)j/ny+(double)k/nz, high=(rand()%1000)/1000.0-0.5;
        hr[id]=low*0.5+high*0.5;
    }
    double *d_p,*d_rhs; cudaMalloc(&d_p,N*8); cudaMalloc(&d_rhs,N*8);

    // reference iteration count (FP64 converge-to-1e-6)
    int it_ref; { CudaPCG3D pcg; cudaMemset(d_p,0,N*8); cudaMemcpy(d_rhs,hr.data(),N*8,cudaMemcpyHostToDevice);
                  pcg.solve_optimized(g,d_p,d_rhs,200,1e-6); it_ref=pcg.last_iters; }

    auto timeit=[&](int mode){   // 0 opt, 1 mixed, 2 device
        CudaPCG3D pcg; double rr=0;
        auto one=[&]{ cudaMemset(d_p,0,N*8); cudaMemcpy(d_rhs,hr.data(),N*8,cudaMemcpyHostToDevice);
            if(mode==0) pcg.solve_optimized(g,d_p,d_rhs,200,1e-6);
            else if(mode==1) pcg.solve_mixed(g,d_p,d_rhs,200,1e-6);
            else pcg.solve_device(g,d_p,d_rhs,it_ref,1e-6); };
        for(int w=0;w<3;w++) one();
        cudaDeviceSynchronize();
        const int R=5; auto t0=std::chrono::high_resolution_clock::now();
        for(int r=0;r<R;r++) one();
        cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        rr=pcg.last_rel_res;
        return std::pair<double,double>(std::chrono::duration<double>(t1-t0).count()*1000.0/R, rr);
    };
    auto o=timeit(0); auto m=timeit(1); auto d=timeit(2);
    printf("  %3dx%3dx%3d (%d it): FP64 %.2f ms | mixed %.2f ms (%.2fx) | device %.2f ms (%.2fx)  [dev ||r||=%.2e]\n",
           nx,ny,nz,it_ref, o.first, m.first,o.first/m.first, d.first,o.first/d.first, d.second);
    cudaFree(d_p); cudaFree(d_rhs); g.free();
}

int main(){
    printf("=== PCG variants vs FP64 baseline (RTX 3090), 5-run avg ===\n");
    run(64,32,32); run(128,64,64); run(256,128,128);
    return 0;
}

/**
 * @file prof_vcycle.cu
 * @brief Minimal target for ncu profiling of the finest-level V-cycle kernels
 *        at 256x128x128. setupLevels once, then a few vcycle_apply; the first
 *        rbgs/restrict/prolong launches are the finest level (the hot kernels).
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"
#include <vector>

int main(int argc, char** argv){
    bool use_float = (argc>1 && argv[1][0]=='f');
    int nx=256,ny=128,nz=128, N=(nx+2)*(ny+2)*(nz+2);
    if(!use_float){
        CudaGrid3D g; g.allocate(nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
        std::vector<double> hr(N); for(int i=0;i<N;i++) hr[i]=(i%97)*0.01-0.5;
        double *r,*z; cudaMalloc(&r,N*8); cudaMalloc(&z,N*8);
        cudaMemcpy(r,hr.data(),N*8,cudaMemcpyHostToDevice);
        CudaUAAMGPreconditioner3D p; p.setupLevels(g);
        for(int i=0;i<6;i++) p.vcycle_apply(g,r,z);
        cudaDeviceSynchronize(); cudaFree(r); cudaFree(z); g.free();
    } else {
        CudaGrid3Df gf; gf.allocate(nx,ny,nz,1.0f/nx,1.0f/ny,1.0f/nz);
        std::vector<float> hrf(N); for(int i=0;i<N;i++) hrf[i]=(i%97)*0.01f-0.5f;
        float *rf,*zf; cudaMalloc(&rf,N*4); cudaMalloc(&zf,N*4);
        cudaMemcpy(rf,hrf.data(),N*4,cudaMemcpyHostToDevice);
        CudaUAAMGPreconditioner3Dt pf; pf.setupLevels(gf);
        for(int i=0;i<6;i++) pf.vcycle_apply(gf,rf,zf);
        cudaDeviceSynchronize(); cudaFree(rf); cudaFree(zf); gf.free();
    }
    return 0;
}

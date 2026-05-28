/**
 * @file test_tiled_soa.cu
 * @brief §5.2 microbenchmark: does an 8³ tiled-SoA memory layout speed up the
 *        RBGS smoother (the V-cycle hot kernel) vs plain column-major?
 *
 * Isolates the memory-layout effect: identical constant-coefficient red-black
 * Gauss-Seidel (the dominant cost = 7 x-reads/cell), one in plain layout
 * x[i + j*nx + k*nx*ny], one in tiled layout (8³ tiles contiguous, bit-shift
 * index). Same work, same grid; the only difference is the layout.
 */
#include <cstdio>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define TB 8

// plain column-major (0-based interior, no halo; neighbours clamped/mirrored)
__device__ inline int pidx(int i,int j,int k,int nx,int ny){ return i + j*nx + k*nx*ny; }

// 8³ tiled: tile (i>>3,j>>3,k>>3), within (i&7,j&7,k&7); bit-shift index
__device__ inline int tidx(int i,int j,int k,int ntx,int nty){
    int tx=i>>3, ty=j>>3, tz=k>>3, wx=i&7, wy=j&7, wz=k&7;
    return (((tz*nty + ty)*ntx + tx)<<9) | (wz<<6) | (wy<<3) | wx;   // *512 + wz*64 + wy*8 + wx
}

__global__ void rbgs_plain(double* x, const double* b, int nx,int ny,int nz, int parity){
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y, k=blockIdx.z*blockDim.z+threadIdx.z;
    if(i>=nx||j>=ny||k>=nz) return;
    if(((i+j+k)&1)!=parity) return;
    int id=pidx(i,j,k,nx,ny); double c=x[id];
    double s = (i>0?x[pidx(i-1,j,k,nx,ny)]:c) + (i<nx-1?x[pidx(i+1,j,k,nx,ny)]:c)
             + (j>0?x[pidx(i,j-1,k,nx,ny)]:c) + (j<ny-1?x[pidx(i,j+1,k,nx,ny)]:c)
             + (k>0?x[pidx(i,j,k-1,nx,ny)]:c) + (k<nz-1?x[pidx(i,j,k+1,nx,ny)]:c);
    x[id] = (b[id] + s) / 6.0;
}

__global__ void rbgs_tiled(double* x, const double* b, int nx,int ny,int nz, int ntx,int nty, int parity){
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y, k=blockIdx.z*blockDim.z+threadIdx.z;
    if(i>=nx||j>=ny||k>=nz) return;
    if(((i+j+k)&1)!=parity) return;
    int id=tidx(i,j,k,ntx,nty); double c=x[id];
    double s = (i>0?x[tidx(i-1,j,k,ntx,nty)]:c) + (i<nx-1?x[tidx(i+1,j,k,ntx,nty)]:c)
             + (j>0?x[tidx(i,j-1,k,ntx,nty)]:c) + (j<ny-1?x[tidx(i,j+1,k,ntx,nty)]:c)
             + (k>0?x[tidx(i,j,k-1,ntx,nty)]:c) + (k<nz-1?x[tidx(i,j,k+1,ntx,nty)]:c);
    x[id] = (b[id] + s) / 6.0;
}

static void bench(int nx,int ny,int nz){
    int N = nx*ny*nz;                 // multiples of 8 ⇒ tiled needs no padding
    int ntx=nx/8, nty=ny/8;
    double *xp,*bp,*xt,*bt;
    cudaMalloc(&xp,N*8); cudaMalloc(&bp,N*8); cudaMalloc(&xt,N*8); cudaMalloc(&bt,N*8);
    cudaMemset(xp,0,N*8); cudaMemset(xt,0,N*8); cudaMemset(bp,0,N*8); cudaMemset(bt,0,N*8);
    dim3 blk(8,8,8), grd((nx+7)/8,(ny+7)/8,(nz+7)/8);
    const int SW=100;
    // plain
    for(int w=0;w<10;w++){ rbgs_plain<<<grd,blk>>>(xp,bp,nx,ny,nz,1); rbgs_plain<<<grd,blk>>>(xp,bp,nx,ny,nz,0); }
    cudaDeviceSynchronize();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int s=0;s<SW;s++){ rbgs_plain<<<grd,blk>>>(xp,bp,nx,ny,nz,1); rbgs_plain<<<grd,blk>>>(xp,bp,nx,ny,nz,0); }
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    double mp=std::chrono::duration<double>(t1-t0).count()*1000.0/SW;
    // tiled
    for(int w=0;w<10;w++){ rbgs_tiled<<<grd,blk>>>(xt,bt,nx,ny,nz,ntx,nty,1); rbgs_tiled<<<grd,blk>>>(xt,bt,nx,ny,nz,ntx,nty,0); }
    cudaDeviceSynchronize();
    auto t2=std::chrono::high_resolution_clock::now();
    for(int s=0;s<SW;s++){ rbgs_tiled<<<grd,blk>>>(xt,bt,nx,ny,nz,ntx,nty,1); rbgs_tiled<<<grd,blk>>>(xt,bt,nx,ny,nz,ntx,nty,0); }
    cudaDeviceSynchronize();
    auto t3=std::chrono::high_resolution_clock::now();
    double mt=std::chrono::duration<double>(t3-t2).count()*1000.0/SW;
    printf("  %3dx%3dx%3d : plain %.3f ms  tiled-SoA %.3f ms  speedup %.2fx\n", nx,ny,nz, mp, mt, mp/mt);
    cudaFree(xp);cudaFree(bp);cudaFree(xt);cudaFree(bt);
}

int main(){
    printf("=== §5.2 layout microbench: RBGS sweep, plain vs 8^3 tiled-SoA (FP64, RTX 3090) ===\n");
    bench(64,32,32); bench(128,64,64); bench(256,128,128); bench(256,256,256);
    return 0;
}

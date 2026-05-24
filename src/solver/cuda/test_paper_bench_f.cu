/**
 * @file test_paper_bench_f.cu
 * @brief FP32 paper-scale benchmark — float PCG + UAAMG preconditioner
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"
#include "solver/cuda/cuda_pcg_3d.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

__device__ inline int fidx(int i,int j,int k,int pitch,int ny){return i+j*pitch+k*pitch*(ny+2);}

#define FT 8
#define FTH (FT+2)

// ── Shared-memory tiled matvec + fused dot(p,Ap) — float ──
__global__ __launch_bounds__(512,2) void matvec_tiled_dot_kernel_f(
    const float * __restrict p, float * __restrict Ap, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch,
    float idx2, float idy2, float idz2, float diag, float * __restrict dot_buf)
{
    __shared__ float sp[FTH][FTH][FTH];
    __shared__ bool  ss[FTH][FTH][FTH];
    __shared__ float sdot[FT*FT*FT];
    int tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;
    int gi=blockIdx.x*FT+tx+1,gj=blockIdx.y*FT+ty+1,gk=blockIdx.z*FT+tz+1;
    int li=tx+1,lj=ty+1,lk=tz+1;
    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    if(valid){int gid=fidx(gi,gj,gk,pitch,ny);sp[li][lj][lk]=p[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sp[li][lj][lk]=0.0f;}
    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=fidx(gl,gj,gk,pitch,ny);sp[0][lj][lk]=p[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sp[0][lj][lk]=0.0f;}}
    if(tx==FT-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=fidx(gr,gj,gk,pitch,ny);sp[FT+1][lj][lk]=p[h];ss[FT+1][lj][lk]=solid[h];}else{ss[FT+1][lj][lk]=true;sp[FT+1][lj][lk]=0.0f;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=fidx(gi,gb,gk,pitch,ny);sp[li][0][lk]=p[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sp[li][0][lk]=0.0f;}}
    if(ty==FT-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=fidx(gi,gt,gk,pitch,ny);sp[li][FT+1][lk]=p[h];ss[li][FT+1][lk]=solid[h];}else{ss[li][FT+1][lk]=true;sp[li][FT+1][lk]=0.0f;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=fidx(gi,gj,gf,pitch,ny);sp[li][lj][0]=p[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sp[li][lj][0]=0.0f;}}
    if(tz==FT-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=fidx(gi,gj,gk2,pitch,ny);sp[li][lj][FT+1]=p[h];ss[li][lj][FT+1]=solid[h];}else{ss[li][lj][FT+1]=true;sp[li][lj][FT+1]=0.0f;}}
    __syncthreads();
    float my_dot=0.0f,Ap_val=0.0f;
    if(valid&&!ss[li][lj][lk]){
        float pC=sp[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sp[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sp[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sp[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sp[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sp[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sp[li][lj][lk+1];
        Ap_val=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
        int gid=fidx(gi,gj,gk,pitch,ny);Ap[gid]=Ap_val;my_dot=pC*Ap_val;
    }
    int tid=tx+ty*FT+tz*(FT*FT);sdot[tid]=my_dot;__syncthreads();
    for(int s=(FT*FT*FT)/2;s>0;s>>=1){if(tid<s)sdot[tid]+=sdot[tid+s];__syncthreads();}
    if(tid==0){int bid=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;dot_buf[bid]=sdot[0];}
}

// ── Fused axpy + dot(r,r) — float ──
__global__ __launch_bounds__(512,2) void axpy_dot_kernel_f(
    float * __restrict y, const float * __restrict x, float a, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch, float * __restrict dot_buf)
{
    __shared__ float sdot[FT*FT*FT];
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    float my_dot=0.0f;
    if(i<=nx&&j<=ny&&k<=nz){int id=fidx(i,j,k,pitch,ny);if(!solid[id]){y[id]+=a*x[id];my_dot=y[id]*y[id];}}
    int tid=threadIdx.x+threadIdx.y*FT+threadIdx.z*(FT*FT);sdot[tid]=my_dot;__syncthreads();
    for(int s=(FT*FT*FT)/2;s>0;s>>=1){if(tid<s)sdot[tid]+=sdot[tid+s];__syncthreads();}
    if(tid==0){int bid=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;dot_buf[bid]=sdot[0];}
}

__global__ void dot_kernel_f(const float *a,const float *b,const bool *solid,int N,float *part){
    __shared__ float s[256];int tid=threadIdx.x;float sum=0;
    for(int k=blockIdx.x*blockDim.x+tid;k<N;k+=blockDim.x*gridDim.x)if(!solid[k])sum+=a[k]*b[k];
    s[tid]=sum;__syncthreads();for(int st=128;st>0;st>>=1){if(tid<st)s[tid]+=s[tid+st];__syncthreads();}
    if(tid==0)part[blockIdx.x]=s[0];
}
__global__ void axpy_kernel_f(float *y,const float *x,float a,const bool *solid,int nx,int ny,int nz,int pitch){
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz)return;int id=fidx(i,j,k,pitch,ny);if(!solid[id])y[id]+=a*x[id];
}
__global__ void negate_kernel_f(float *v,const bool *solid,int nx,int ny,int nz,int pitch){
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz)return;int id=fidx(i,j,k,pitch,ny);if(!solid[id])v[id]=-v[id];
}
__global__ void submean_kernel_f(float *v,float m,const bool *solid,int nx,int ny,int nz,int pitch){
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>nx||j>ny||k>nz)return;int id=fidx(i,j,k,pitch,ny);if(!solid[id])v[id]-=m;
}
__global__ void sum_interior_kernel_f(const float *v,const bool *solid,int nx,int ny,int nz,int pitch,float *part){
    __shared__ float s[256];int tid=threadIdx.x;float sum=0;
    for(int lin=blockIdx.x*blockDim.x+tid;lin<nx*ny*nz;lin+=blockDim.x*gridDim.x){
        int i=(lin%nx)+1,j=((lin/nx)%ny)+1,k=(lin/(nx*ny))+1;int id=fidx(i,j,k,pitch,ny);if(!solid[id])sum+=v[id];
    }
    s[tid]=sum;__syncthreads();for(int st=128;st>0;st>>=1){if(tid<st)s[tid]+=s[tid+st];__syncthreads();}
    if(tid==0)part[blockIdx.x]=s[0];
}
__global__ void count_interior_kernel_f(const bool *solid,int nx,int ny,int nz,int pitch,int *part){
    __shared__ int s[256];int tid=threadIdx.x;int sum=0;
    for(int lin=blockIdx.x*blockDim.x+tid;lin<nx*ny*nz;lin+=blockDim.x*gridDim.x){
        int i=(lin%nx)+1,j=((lin/nx)%ny)+1,k=(lin/(nx*ny))+1;if(!solid[fidx(i,j,k,pitch,ny)])sum++;
    }
    s[tid]=sum;__syncthreads();for(int st=128;st>0;st>>=1){if(tid<st)s[tid]+=s[tid+st];__syncthreads();}
    if(tid==0)part[blockIdx.x]=s[0];
}

static float host_sum_f(const float *d,int n){std::vector<float> h(n);cudaMemcpy(h.data(),d,n*sizeof(float),cudaMemcpyDeviceToHost);float t=0;for(float v:h)t+=v;return t;}
static int host_sum_int_f(const int *d,int n){std::vector<int> h(n);cudaMemcpy(h.data(),d,n*sizeof(int),cudaMemcpyDeviceToHost);int t=0;for(int v:h)t+=v;return t;}
static float gpu_mean_f(const float *v,const bool *solid,int nx,int ny,int nz,int pitch,float *part,int *cnt,int nb){
    sum_interior_kernel_f<<<nb,256>>>(v,solid,nx,ny,nz,pitch,part);cudaDeviceSynchronize();
    float s=host_sum_f(part,nb);count_interior_kernel_f<<<nb,256>>>(solid,nx,ny,nz,pitch,cnt);cudaDeviceSynchronize();
    int c=host_sum_int_f(cnt,nb);return c>0?s/c:0.0f;
}

// ═══════════════════════════════════════════════════════════════
struct CudaPCG3Df {
    CudaUAAMGPreconditioner3Df precond_;
    float *d_r=nullptr,*d_z=nullptr,*d_p=nullptr,*d_Ap=nullptr,*d_dot=nullptr;
    int *d_cnt=nullptr;int dot_buf_sz_=0,N_=0;
    void ensure(int N){
        if(N_>=N)return;
        if(d_r)cudaFree(d_r);if(d_z)cudaFree(d_z);if(d_p)cudaFree(d_p);if(d_Ap)cudaFree(d_Ap);
        if(d_dot)cudaFree(d_dot);if(d_cnt)cudaFree(d_cnt);
        cudaMalloc(&d_r,N*sizeof(float));cudaMalloc(&d_z,N*sizeof(float));
        cudaMalloc(&d_p,N*sizeof(float));cudaMalloc(&d_Ap,N*sizeof(float));
        int mb=(N+255)/256+1;cudaMalloc(&d_dot,mb*sizeof(float));cudaMalloc(&d_cnt,mb*sizeof(int));
        dot_buf_sz_=mb;N_=N;
    }
    ~CudaPCG3Df(){if(d_r)cudaFree(d_r);if(d_z)cudaFree(d_z);if(d_p)cudaFree(d_p);if(d_Ap)cudaFree(d_Ap);if(d_dot)cudaFree(d_dot);if(d_cnt)cudaFree(d_cnt);}
    float solve_optimized(CudaGrid3Df& g,float* p,float* rhs,int max_iter,float tol);
};

float CudaPCG3Df::solve_optimized(CudaGrid3Df& g,float* p,float* rhs,int max_iter,float tol){
    int nx=g.nx,ny=g.ny,nz=g.nz,pitch=g.pitch,N=(nx+2)*(ny+2)*(nz+2);
    ensure(N);dim3 block3d(FT,FT,FT),grid3d((nx+FT-1)/FT,(ny+FT-1)/FT,(nz+FT-1)/FT);
    int n3d=grid3d.x*grid3d.y*grid3d.z,nb1d=dot_buf_sz_;
    precond_.build(g);
    cudaMemcpy(d_r,rhs,N*sizeof(float),cudaMemcpyDeviceToDevice);cudaMemset(p,0,N*sizeof(float));
    float mr=gpu_mean_f(d_r,g.solid,nx,ny,nz,pitch,d_dot,d_cnt,nb1d);
    submean_kernel_f<<<grid3d,block3d>>>(d_r,mr,g.solid,nx,ny,nz,pitch);
    negate_kernel_f<<<grid3d,block3d>>>(d_r,g.solid,nx,ny,nz,pitch);
    precond_.apply_optimized(g,d_r,d_z);
    float mz=gpu_mean_f(d_z,g.solid,nx,ny,nz,pitch,d_dot,d_cnt,nb1d);
    submean_kernel_f<<<grid3d,block3d>>>(d_z,mz,g.solid,nx,ny,nz,pitch);
    cudaMemcpy(d_p,d_z,N*sizeof(float),cudaMemcpyDeviceToDevice);
    dot_kernel_f<<<nb1d,256>>>(d_r,d_z,g.solid,N,d_dot);cudaDeviceSynchronize();
    float rsold=host_sum_f(d_dot,nb1d);
    if(rsold<1e-30f){cudaMemset(p,0,N*sizeof(float));return 0;}
    for(int k=0;k<max_iter;k++){
        matvec_tiled_dot_kernel_f<<<grid3d,block3d>>>(d_p,d_Ap,g.solid,nx,ny,nz,pitch,g.idx2,g.idy2,g.idz2,g.diag,d_dot);
        cudaDeviceSynchronize();float pAp=host_sum_f(d_dot,n3d);
        if(pAp<1e-15f)break;
        float alpha=rsold/pAp;
        axpy_kernel_f<<<grid3d,block3d>>>(p,d_p,alpha,g.solid,nx,ny,nz,pitch);
        axpy_dot_kernel_f<<<grid3d,block3d>>>(d_r,d_Ap,-alpha,g.solid,nx,ny,nz,pitch,d_dot);
        cudaDeviceSynchronize();float rsnew=host_sum_f(d_dot,n3d);
        if(sqrtf(rsnew)<tol)break;
        precond_.apply_optimized(g,d_r,d_z);
        mz=gpu_mean_f(d_z,g.solid,nx,ny,nz,pitch,d_dot,d_cnt,nb1d);
        submean_kernel_f<<<grid3d,block3d>>>(d_z,mz,g.solid,nx,ny,nz,pitch);cudaDeviceSynchronize();
        dot_kernel_f<<<nb1d,256>>>(d_r,d_z,g.solid,N,d_dot);cudaDeviceSynchronize();
        float rz=host_sum_f(d_dot,nb1d),beta=rz/rsold;rsold=rz;
        cudaMemcpy(d_Ap,d_p,N*sizeof(float),cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p,d_z,N*sizeof(float),cudaMemcpyDeviceToDevice);
        axpy_kernel_f<<<grid3d,block3d>>>(d_p,d_Ap,beta,g.solid,nx,ny,nz,pitch);cudaDeviceSynchronize();
    }
    return rsold;
}

// ═══════════════════════════════════════════════════════════════
int main(){
    printf("============================================================\n");
    printf("  UAAMGPCG FP32 vs FP64 Benchmark\n");
    printf("  Paper: 256x128x128, RTX 4090, V(1,1)\n");
    printf("  Ours:  RTX 3090, V(1,1)\n");
    printf("============================================================\n\n");
    struct Bench{int nx,ny,nz,iters;};
    std::vector<Bench> grids={{64,32,32,20},{128,64,64,30},{256,128,128,40}};
    for(auto& B:grids){
        int nx=B.nx,ny=B.ny,nz=B.nz,N=(nx+2)*(ny+2)*(nz+2);
        printf("═══ Grid %dx%dx%d (%.1fM cells) ═══\n",nx,ny,nz,nx*ny*nz/1e6);
        std::vector<float> h_rhs(N,0.0f);std::vector<char> h_solid(N,0);
        for(int i=1;i<=nx;i++)for(int j=1;j<=ny;j++)for(int k=1;k<=nz;k++){
            int id=i+j*(nx+2)+k*(nx+2)*(ny+2);h_rhs[id]=((float)i/nx+(float)j/ny+(float)k/nz)*0.5f+((rand()%1000)/1000.0f-0.5f)*0.5f;
        }
        // Double opt
        CudaGrid3D gd;gd.allocate(nx,ny,nz,1.0/nx,1.0/ny,1.0/nz);
        cudaMemcpy(gd.solid,h_solid.data(),N*sizeof(bool),cudaMemcpyHostToDevice);
        double *dp,*dr;cudaMalloc(&dp,N*sizeof(double));cudaMalloc(&dr,N*sizeof(double));
        std::vector<double> hrd(N);for(int i=0;i<N;i++)hrd[i]=h_rhs[i];
        cudaMemcpy(dr,hrd.data(),N*sizeof(double),cudaMemcpyHostToDevice);
        CudaPCG3D pcg;
        for(int w=0;w<2;w++){cudaMemset(dp,0,N*sizeof(double));pcg.solve_optimized(gd,dp,dr,5,1e-6f);}cudaDeviceSynchronize();
        cudaMemset(dp,0,N*sizeof(double));cudaMemcpy(dr,hrd.data(),N*sizeof(double),cudaMemcpyHostToDevice);
        auto t0=std::chrono::high_resolution_clock::now();
        pcg.solve_optimized(gd,dp,dr,B.iters,1e-6f);cudaDeviceSynchronize();
        auto t1=std::chrono::high_resolution_clock::now();
        double d_ms=std::chrono::duration<double>(t1-t0).count()*1000.0;
        // Float opt
        CudaGrid3Df gf;gf.allocate(nx,ny,nz,1.0f/nx,1.0f/ny,1.0f/nz);
        cudaMemcpy(gf.solid,h_solid.data(),N*sizeof(bool),cudaMemcpyHostToDevice);
        float *fp,*fr;cudaMalloc(&fp,N*sizeof(float));cudaMalloc(&fr,N*sizeof(float));
        cudaMemcpy(fr,h_rhs.data(),N*sizeof(float),cudaMemcpyHostToDevice);
        CudaPCG3Df pcgf;
        for(int w=0;w<2;w++){cudaMemset(fp,0,N*sizeof(float));pcgf.solve_optimized(gf,fp,fr,5,1e-6f);}cudaDeviceSynchronize();
        cudaMemset(fp,0,N*sizeof(float));cudaMemcpy(fr,h_rhs.data(),N*sizeof(float),cudaMemcpyHostToDevice);
        auto t2=std::chrono::high_resolution_clock::now();
        pcgf.solve_optimized(gf,fp,fr,B.iters,1e-6f);cudaDeviceSynchronize();
        auto t3=std::chrono::high_resolution_clock::now();
        double f_ms=std::chrono::duration<double>(t3-t2).count()*1000.0;
        printf("  Double-opt: %8.2f ms\n",d_ms);
        printf("  Float-opt:  %8.2f ms  speedup=%.2fx\n",f_ms,d_ms/f_ms);
        cudaFree(dp);cudaFree(dr);cudaFree(fp);cudaFree(fr);gd.free();gf.free();printf("\n");
    }
    printf("============================================================\n");
    return 0;
}

/**
 * @file cuda_uaamg_preconditioner_3d_opt_f.cu
 * @brief FP32 UAAMG paper-optimized 3D V-cycle (SIGGRAPH 2025 Section 5.3).
 *
 * Float-precision counterpart of cuda_uaamg_preconditioner_3d_opt.cu.
 * Same shared-memory tiling + aggregated kernel optimizations.
 *
 * Optimizations (cumulative):
 *   1. Shared-memory tiled RBGS — 8^3 tile + halo, ~6x fewer global reads
 *   2. Aggregated pre-smooth+Restrict — one kernel launch
 *   3. Aggregated Prolong+post-smooth — one kernel launch
 *   4. Fused coarsest solves — 20 sweeps in 1 launch
 *   5. Redundant zero-x eliminated — x zeroed once upfront
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"

__device__ inline int opti_idx_f(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

#define TILED_T 8
#define TILED_TH (TILED_T + 2)

// ═══════════════════════════════════════════════════════════════
//  Shared-memory tiled RBGS (one sweep: red+black in 1 launch)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_tiled_kernel_f(
    float * __restrict x, const float * __restrict b, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch,
    float idx2, float idy2, float idz2, float diag)
{
    __shared__ float sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool  ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx_f(gi,gj,gk,pitch,ny):-1;

    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0f;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx_f(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0f;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx_f(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0f;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx_f(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0f;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx_f(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0f;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx_f(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0f;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx_f(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0f;}}
    __syncthreads();

    float inv_diag=1.0f/diag;
    // Red pass
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Black pass
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    if(valid&&!ss[li][lj][lk])x[gid]=sx[li][lj][lk];
}

// ═══════════════════════════════════════════════════════════════
//  Aggregated RBGS + Restrict (down-sweep: smooth + residual + 8-to-1)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_restrict_aggregated_kernel_f(
    float * __restrict x, const float * __restrict b, const bool * __restrict solid,
    float * __restrict b_coarse, const bool * __restrict solid_coarse,
    int nx, int ny, int nz, int pitch,
    float idx2, float idy2, float idz2, float diag,
    int cnx, int cny, int cpitch)
{
    __shared__ float sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool  ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx_f(gi,gj,gk,pitch,ny):-1;

    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0f;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx_f(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0f;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx_f(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0f;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx_f(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0f;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx_f(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0f;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx_f(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0f;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx_f(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0f;}}
    __syncthreads();

    float inv_diag=1.0f/diag;
    // Red pass
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Black pass
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Restrict + write-back
    if(valid&&!ss[li][lj][lk]){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk];
        float pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk];
        float pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1];
        float pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float res=b[gid]-diag*pC+lap;
        int ic=(gi+1)/2,jc=(gj+1)/2,kc=(gk+1)/2;
        int cid=opti_idx_f(ic,jc,kc,cpitch,cny);
        if(!solid_coarse[cid])atomicAdd(&b_coarse[cid],res/8.0f);
        x[gid]=sx[li][lj][lk];
    }
}

// ═══════════════════════════════════════════════════════════════
//  Aggregated Prolong + Post-smooth (up-sweep: correct + smooth)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void prolong_rbgs_aggregated_kernel_f(
    float * __restrict x_fine, const float * __restrict x_coarse, const float * __restrict b, const bool * __restrict solid,
    int fnx, int fny, int fnz, int fpitch,
    float idx2, float idy2, float idz2, float diag,
    int cny, int cpitch)
{
    __shared__ float sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool  ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=fnx&&gj<=fny&&gk<=fnz);
    int gid=valid?opti_idx_f(gi,gj,gk,fpitch,fny):-1;

    if(valid){sx[li][lj][lk]=x_fine[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0f;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=fny&&gk<=fnz){int h=opti_idx_f(gl,gj,gk,fpitch,fny);sx[0][lj][lk]=x_fine[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0f;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=fnx&&gj<=fny&&gk<=fnz){int h=opti_idx_f(gr,gj,gk,fpitch,fny);sx[TILED_T+1][lj][lk]=x_fine[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0f;}}
    if(ty==0){int gb=gj-1;if(gi<=fnx&&gb>=1&&gk<=fnz){int h=opti_idx_f(gi,gb,gk,fpitch,fny);sx[li][0][lk]=x_fine[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0f;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=fnx&&gt<=fny&&gk<=fnz){int h=opti_idx_f(gi,gt,gk,fpitch,fny);sx[li][TILED_T+1][lk]=x_fine[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0f;}}
    if(tz==0){int gf=gk-1;if(gi<=fnx&&gj<=fny&&gf>=1){int h=opti_idx_f(gi,gj,gf,fpitch,fny);sx[li][lj][0]=x_fine[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0f;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=fnx&&gj<=fny&&gk2<=fnz){int h=opti_idx_f(gi,gj,gk2,fpitch,fny);sx[li][lj][TILED_T+1]=x_fine[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0f;}}
    __syncthreads();

    // Prolong correction in shared memory
    if(valid&&!ss[li][lj][lk]){
        int ic=(gi+1)/2,jc=(gj+1)/2,kc=(gk+1)/2;
        sx[li][lj][lk]+=2.0f*x_coarse[opti_idx_f(ic,jc,kc,cpitch,cny)];
    }
    __syncthreads();

    float inv_diag=1.0f/diag;
    // Red pass
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Black pass
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        float pC=sx[li][lj][lk];
        float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        float ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    if(valid&&!ss[li][lj][lk])x_fine[gid]=sx[li][lj][lk];
}

// ═══════════════════════════════════════════════════════════════
//  Coarsest level: 20 sweeps in 1 launch (b cached in register)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_coarsest_kernel_f(
    float * __restrict x, const float * __restrict b, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch,
    float idx2, float idy2, float idz2, float diag)
{
    __shared__ float sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool  ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx_f(gi,gj,gk,pitch,ny):-1;
    float my_b=valid?b[gid]:0.0f;

    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0f;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx_f(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0f;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx_f(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0f;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx_f(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0f;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx_f(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0f;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx_f(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0f;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx_f(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0f;}}
    __syncthreads();

    float inv_diag=1.0f/diag;
    for(int sweep=0;sweep<20;sweep++){
        if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
            float pC=sx[li][lj][lk];
            float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
            float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
            float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
            float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
            float ed=diag;
            if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
            if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
            if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
            float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
            sx[li][lj][lk]+=inv*(my_b-diag*pC+lap);
        }
        __syncthreads();
        if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
            float pC=sx[li][lj][lk];
            float pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
            float pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
            float pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
            float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
            float ed=diag;
            if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
            if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
            if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
            float inv=(ed==diag)?inv_diag:((ed<1e-15f)?0.0f:1.0f/ed);
            sx[li][lj][lk]+=inv*(my_b-diag*pC+lap);
        }
        __syncthreads();
    }

    if(valid&&!ss[li][lj][lk])x[gid]=sx[li][lj][lk];
}

// ── Utility kernels ──
__global__ void restrict_opt_kernel_f(
    const float *x, const float *b, const bool *solid,
    float *b_coarse, bool *solid_coarse,
    int fnx, int fny, int fnz, int fpitch,
    float idx2, float idy2, float idz2, float diag,
    int cstride)
{
    int ic=blockIdx.x*blockDim.x+threadIdx.x+1,jc=blockIdx.y*blockDim.y+threadIdx.y+1,kc=blockIdx.z*blockDim.z+threadIdx.z+1;
    int cnx=fnx/2,cny=fny/2,cnz=fnz/2;
    if(ic>cnx||jc>cny||kc>cnz)return;
    int i_f=2*ic-1,j_f=2*jc-1,k_f=2*kc-1;float sum=0;int cnt=0;
    for(int di=0;di<2;di++)for(int dj=0;dj<2;dj++)for(int dk=0;dk<2;dk++){
        int fi=i_f+di,fj=j_f+dj,fk=k_f+dk;
        int fidx=opti_idx_f(fi,fj,fk,fpitch,fny);
        if(solid[fidx])continue;
        float pC=x[fidx];
        float pL=(fi>1&&!solid[opti_idx_f(fi-1,fj,fk,fpitch,fny)])?x[opti_idx_f(fi-1,fj,fk,fpitch,fny)]:pC;
        float pR=(fi<fnx&&!solid[opti_idx_f(fi+1,fj,fk,fpitch,fny)])?x[opti_idx_f(fi+1,fj,fk,fpitch,fny)]:pC;
        float pB=(fj>1&&!solid[opti_idx_f(fi,fj-1,fk,fpitch,fny)])?x[opti_idx_f(fi,fj-1,fk,fpitch,fny)]:pC;
        float pT=(fj<fny&&!solid[opti_idx_f(fi,fj+1,fk,fpitch,fny)])?x[opti_idx_f(fi,fj+1,fk,fpitch,fny)]:pC;
        float pF=(fk>1&&!solid[opti_idx_f(fi,fj,fk-1,fpitch,fny)])?x[opti_idx_f(fi,fj,fk-1,fpitch,fny)]:pC;
        float pK=(fk<fnz&&!solid[opti_idx_f(fi,fj,fk+1,fpitch,fny)])?x[opti_idx_f(fi,fj,fk+1,fpitch,fny)]:pC;
        float lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        sum+=b[fidx]-diag*pC+lap;cnt++;
    }
    int cid=opti_idx_f(ic,jc,kc,cstride,cny);
    if(!solid_coarse[cid]&&cnt>0)b_coarse[cid]=sum/cnt;
}

__global__ void prolong_opt_kernel_f(
    float *x_fine, const float *x_coarse, const bool *solid_fine,
    int fnx, int fny, int fnz, int fpitch, int cstride)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>fnx||j>fny||k>fnz)return;
    int fid=opti_idx_f(i,j,k,fpitch,fny);
    if(solid_fine[fid])return;
    int ic=(i+1)/2,jc=(j+1)/2,kc=(k+1)/2;
    x_fine[fid]+=2.0f*x_coarse[opti_idx_f(ic,jc,kc,cstride,fny/2)];
}

__global__ void zero_kernel_opt_f(float *a, int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)a[i]=0.0f;}
__global__ void copy_kernel_opt_f(float *dst, const float *src, int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)dst[i]=src[i];}
__global__ void restrict_solid_opt_kernel_f(
    const bool *sf, bool *sc, int fnx, int fny, int fnz, int fpitch,
    int cnx, int cny, int cnz, int cpitch)
{
    int ic=blockIdx.x*blockDim.x+threadIdx.x+1,jc=blockIdx.y*blockDim.y+threadIdx.y+1,kc=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ic>cnx||jc>cny||kc>cnz)return;
    int i_f=2*ic-1,j_f=2*jc-1,k_f=2*kc-1,scount=0;
    for(int di=0;di<2;di++)for(int dj=0;dj<2;dj++)for(int dk=0;dk<2;dk++)
        if(sf[opti_idx_f(i_f+di,j_f+dj,k_f+dk,fpitch,fny)])scount++;
    sc[opti_idx_f(ic,jc,kc,cpitch,cny)]=(scount>=4);
}

// ═══════════════════════════════════════════════════════════════
//  Optimized V-Cycle for float
// ═══════════════════════════════════════════════════════════════
static void vCycle_opt_f(CudaUAAMGPreconditioner3Df::Level* levels, int lv, int nl, cudaStream_t stream) {
    auto& L = levels[lv];
    int nx=L.g.nx, ny=L.g.ny, nz=L.g.nz;
    dim3 block(TILED_T,TILED_T,TILED_T);
    dim3 grid((nx+TILED_T-1)/TILED_T,(ny+TILED_T-1)/TILED_T,(nz+TILED_T-1)/TILED_T);

    if(lv==nl-1){
        rbgs_coarsest_kernel_f<<<grid,block,0,stream>>>(L.g.x,L.g.b,L.g.solid,nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag);
        return;
    }

    auto& coarse=levels[lv+1];
    int cnx=coarse.g.nx,cny=coarse.g.ny,cnz=coarse.g.nz;
    int Nc=(cnx+2)*(cny+2)*(cnz+2);

    cudaMemsetAsync(coarse.g.b, 0, Nc*sizeof(float), stream);
    rbgs_restrict_aggregated_kernel_f<<<grid,block,0,stream>>>(
        L.g.x,L.g.b,L.g.solid,coarse.g.b,coarse.g.solid,
        nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag,
        cnx,cny,coarse.g.pitch);

    vCycle_opt_f(levels,lv+1,nl,stream);

    prolong_rbgs_aggregated_kernel_f<<<grid,block,0,stream>>>(
        L.g.x,coarse.g.x,L.g.b,L.g.solid,
        nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag,
        cny,coarse.g.pitch);
}

// ═══════════════════════════════════════════════════════════════
//  CudaUAAMGPreconditioner3Df implementation
// ═══════════════════════════════════════════════════════════════
void CudaUAAMGPreconditioner3Df::build(const CudaGrid3Df& fine) {
    if (cached_nx_ == fine.nx && cached_ny_ == fine.ny && cached_nz_ == fine.nz) return;
    destroy();
    int nx = fine.nx, ny = fine.ny, nz = fine.nz;
    float dx = fine.dx, dy = fine.dy, dz = fine.dz;
    while (nx >= 2 && ny >= 2 && nz >= 2) {
        Level L;
        L.g.allocate(nx, ny, nz, dx, dy, dz);
        L.stride = nx + 2;
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4) break;
        nx /= 2; ny /= 2; nz /= 2;
        dx *= 2.0f; dy *= 2.0f; dz *= 2.0f;
    }
    cached_nx_ = fine.nx; cached_ny_ = fine.ny; cached_nz_ = fine.nz;
}

void CudaUAAMGPreconditioner3Df::apply_optimized(const CudaGrid3Df& fine, const float* r, float* z) {
    this->build(fine);
    int nl=(int)levels_.size();
    if(nl==0)return;
    cudaStream_t stream=0;
    int N0=(fine.nx+2)*(fine.ny+2)*(fine.nz+2);

    cudaMemcpy(levels_[0].g.solid,fine.solid,N0*sizeof(bool),cudaMemcpyDeviceToDevice);
    for(int l=1;l<nl;l++){
        auto&fL=levels_[l-1],&cL=levels_[l];
        dim3 block(TILED_T,TILED_T,TILED_T);
        dim3 grid((cL.g.nx+TILED_T-1)/TILED_T,(cL.g.ny+TILED_T-1)/TILED_T,(cL.g.nz+TILED_T-1)/TILED_T);
        restrict_solid_opt_kernel_f<<<grid,block,0,stream>>>(
            fL.g.solid,cL.g.solid,fL.g.nx,fL.g.ny,fL.g.nz,fL.g.pitch,
            cL.g.nx,cL.g.ny,cL.g.nz,cL.g.pitch);
    }

    copy_kernel_opt_f<<<(N0+255)/256,256,0,stream>>>(levels_[0].g.b,r,N0);
    for(int l=0;l<nl;l++){
        int N=(levels_[l].g.nx+2)*(levels_[l].g.ny+2)*(levels_[l].g.nz+2);
        cudaMemsetAsync(levels_[l].g.x, 0, N*sizeof(float), stream);
    }

    vCycle_opt_f(levels_.data(),0,nl,stream);
    cudaDeviceSynchronize();

    copy_kernel_opt_f<<<(N0+255)/256,256,0,stream>>>(z,levels_[0].g.x,N0);
    cudaDeviceSynchronize();
}

void CudaUAAMGPreconditioner3Df::destroy() {
    for (auto& L : levels_) L.g.free();
    levels_.clear();
    cached_nx_ = cached_ny_ = cached_nz_ = -1;
}

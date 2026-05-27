/**
 * @file cuda_uaamg_preconditioner_3d_opt.cu
 * @brief UAAMG paper-optimized 3D V-cycle (SIGGRAPH 2025 Section 5.3).
 *
 * Optimizations (cumulative):
 *   1. Shared-memory tiled RBGS — 8^3 tile + halo, ~6x fewer global reads
 *   2. Aggregated pre-smooth+Restrict — one kernel launch, no x write+re-read
 *   3. Aggregated Prolong+post-smooth — one kernel launch, up-sweep
 *   4. Fused coarsest solves — 20 sweeps in 1 launch
 *   5. Redundant zero-x eliminated — x zeroed once upfront, not per level
 *
 * V-cycle kernel count: 4 levels: 2 zero_b + 1 aggr + 1 coarsest + 2 aggr = 6 launches
 *   (vs ~14 in original: 4 pre*2 + 4 restrict + 4 prolong + 4 post*2 + 1 coarsest*20)
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"

__device__ inline int opti_idx(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

#define TILED_T 8
#define TILED_TH (TILED_T + 2)

// ═══════════════════════════════════════════════════════════════
//  Shared-memory tiled RBGS (one sweep: red+black in 1 launch)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_tiled_kernel(
    double * __restrict x, const double * __restrict b, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag)
{
    __shared__ double sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool   ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx(gi,gj,gk,pitch,ny):-1;

    // Load interior
    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0;}

    // Load 6 face halos
    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0;}}
    __syncthreads();

    double inv_diag=1.0/diag; // precompute once for all cells
    // Red pass
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Black pass
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    if(valid&&!ss[li][lj][lk])x[gid]=sx[li][lj][lk];
}

// ═══════════════════════════════════════════════════════════════
//  Smooth-only RBGS — one red + one black pass, no restriction.
//  Used by V(2,2) and V(2,1) cycles where we want an extra smoothing
//  step that is NOT followed by a restriction.
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_smooth_only_kernel(
    double * __restrict x, const double * __restrict b, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag)
{
    __shared__ double sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool   ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;
    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx(gi,gj,gk,pitch,ny):-1;

    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0;}
    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0;}}
    __syncthreads();

    double inv_diag=1.0/diag;
    // Red
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();
    // Black
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    if(valid&&!ss[li][lj][lk])x[gid]=sx[li][lj][lk];
}

// ═══════════════════════════════════════════════════════════════
//  Aggregated RBGS + Restrict (down-sweep: smooth + residual + 8-to-1)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_restrict_aggregated_kernel(
    double * __restrict x, const double * __restrict b, const bool * __restrict solid,
    double * __restrict b_coarse, const bool * __restrict solid_coarse,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag,
    int cnx, int cny, int cpitch)
{
    __shared__ double sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool   ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx(gi,gj,gk,pitch,ny):-1;

    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0;}}
    __syncthreads();

    double inv_diag=1.0/diag;
    // Red pass
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Black pass
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Restrict + write-back
    if(valid&&!ss[li][lj][lk]){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk];
        double pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk];
        double pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1];
        double pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double res=b[gid]-diag*pC+lap;
        int ic=(gi+1)/2,jc=(gj+1)/2,kc=(gk+1)/2;
        int cid=opti_idx(ic,jc,kc,cpitch,cny);
        if(!solid_coarse[cid])atomicAdd(&b_coarse[cid],res/8.0);
        x[gid]=sx[li][lj][lk];
    }
}

// ═══════════════════════════════════════════════════════════════
//  Aggregated Prolong + Post-smooth (up-sweep: correct + smooth)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void prolong_rbgs_aggregated_kernel(
    double * __restrict x_fine, const double * __restrict x_coarse, const double * __restrict b, const bool * __restrict solid,
    int fnx, int fny, int fnz, int fpitch,
    double idx2, double idy2, double idz2, double diag,
    int cny, int cpitch)
{
    __shared__ double sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool   ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=fnx&&gj<=fny&&gk<=fnz);
    int gid=valid?opti_idx(gi,gj,gk,fpitch,fny):-1;

    if(valid){sx[li][lj][lk]=x_fine[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=fny&&gk<=fnz){int h=opti_idx(gl,gj,gk,fpitch,fny);sx[0][lj][lk]=x_fine[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=fnx&&gj<=fny&&gk<=fnz){int h=opti_idx(gr,gj,gk,fpitch,fny);sx[TILED_T+1][lj][lk]=x_fine[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0;}}
    if(ty==0){int gb=gj-1;if(gi<=fnx&&gb>=1&&gk<=fnz){int h=opti_idx(gi,gb,gk,fpitch,fny);sx[li][0][lk]=x_fine[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=fnx&&gt<=fny&&gk<=fnz){int h=opti_idx(gi,gt,gk,fpitch,fny);sx[li][TILED_T+1][lk]=x_fine[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0;}}
    if(tz==0){int gf=gk-1;if(gi<=fnx&&gj<=fny&&gf>=1){int h=opti_idx(gi,gj,gf,fpitch,fny);sx[li][lj][0]=x_fine[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=fnx&&gj<=fny&&gk2<=fnz){int h=opti_idx(gi,gj,gk2,fpitch,fny);sx[li][lj][TILED_T+1]=x_fine[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0;}}
    __syncthreads();

    // Prolong correction in shared memory
    if(valid&&!ss[li][lj][lk]){
        int ic=(gi+1)/2,jc=(gj+1)/2,kc=(gk+1)/2;
        sx[li][lj][lk]+=2.0*x_coarse[opti_idx(ic,jc,kc,cpitch,cny)];
    }
    __syncthreads();

    double inv_diag=1.0/diag;
    // Red pass
    if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    // Black pass
    if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
        double pC=sx[li][lj][lk];
        double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
        double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
        double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        double ed=diag;
        if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
        if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
        if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
        double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
        sx[li][lj][lk]+=inv*(b[gid]-diag*pC+lap);
    }
    __syncthreads();

    if(valid&&!ss[li][lj][lk])x_fine[gid]=sx[li][lj][lk];
}

// ═══════════════════════════════════════════════════════════════
//  Coarsest level: 20 sweeps in 1 launch (b cached in register)
// ═══════════════════════════════════════════════════════════════
__global__ __launch_bounds__(512,2) void rbgs_coarsest_kernel(
    double * __restrict x, const double * __restrict b, const bool * __restrict solid,
    int nx, int ny, int nz, int pitch,
    double idx2, double idy2, double idz2, double diag)
{
    __shared__ double sx[TILED_TH][TILED_TH][TILED_TH];
    __shared__ bool   ss[TILED_TH][TILED_TH][TILED_TH];

    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*TILED_T+tx+1, gj=blockIdx.y*TILED_T+ty+1, gk=blockIdx.z*TILED_T+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;

    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    int gid=valid?opti_idx(gi,gj,gk,pitch,ny):-1;
    double my_b=valid?b[gid]:0.0;

    if(valid){sx[li][lj][lk]=x[gid];ss[li][lj][lk]=solid[gid];}
    else{ss[li][lj][lk]=true;sx[li][lj][lk]=0.0;}

    if(tx==0){int gl=gi-1;if(gl>=1&&gj<=ny&&gk<=nz){int h=opti_idx(gl,gj,gk,pitch,ny);sx[0][lj][lk]=x[h];ss[0][lj][lk]=solid[h];}else{ss[0][lj][lk]=true;sx[0][lj][lk]=0.0;}}
    if(tx==TILED_T-1){int gr=gi+1;if(gr<=nx&&gj<=ny&&gk<=nz){int h=opti_idx(gr,gj,gk,pitch,ny);sx[TILED_T+1][lj][lk]=x[h];ss[TILED_T+1][lj][lk]=solid[h];}else{ss[TILED_T+1][lj][lk]=true;sx[TILED_T+1][lj][lk]=0.0;}}
    if(ty==0){int gb=gj-1;if(gi<=nx&&gb>=1&&gk<=nz){int h=opti_idx(gi,gb,gk,pitch,ny);sx[li][0][lk]=x[h];ss[li][0][lk]=solid[h];}else{ss[li][0][lk]=true;sx[li][0][lk]=0.0;}}
    if(ty==TILED_T-1){int gt=gj+1;if(gi<=nx&&gt<=ny&&gk<=nz){int h=opti_idx(gi,gt,gk,pitch,ny);sx[li][TILED_T+1][lk]=x[h];ss[li][TILED_T+1][lk]=solid[h];}else{ss[li][TILED_T+1][lk]=true;sx[li][TILED_T+1][lk]=0.0;}}
    if(tz==0){int gf=gk-1;if(gi<=nx&&gj<=ny&&gf>=1){int h=opti_idx(gi,gj,gf,pitch,ny);sx[li][lj][0]=x[h];ss[li][lj][0]=solid[h];}else{ss[li][lj][0]=true;sx[li][lj][0]=0.0;}}
    if(tz==TILED_T-1){int gk2=gk+1;if(gi<=nx&&gj<=ny&&gk2<=nz){int h=opti_idx(gi,gj,gk2,pitch,ny);sx[li][lj][TILED_T+1]=x[h];ss[li][lj][TILED_T+1]=solid[h];}else{ss[li][lj][TILED_T+1]=true;sx[li][lj][TILED_T+1]=0.0;}}
    __syncthreads();

    double inv_diag=1.0/diag;
    for(int sweep=0;sweep<20;sweep++){
        if(valid&&!ss[li][lj][lk]&&((gi+gj+gk)&1)){
            double pC=sx[li][lj][lk];
            double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
            double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
            double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
            double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
            double ed=diag;
            if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
            if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
            if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
            double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
            sx[li][lj][lk]+=inv*(my_b-diag*pC+lap);
        }
        __syncthreads();
        if(valid&&!ss[li][lj][lk]&&!((gi+gj+gk)&1)){
            double pC=sx[li][lj][lk];
            double pL=ss[li-1][lj][lk]?pC:sx[li-1][lj][lk],pR=ss[li+1][lj][lk]?pC:sx[li+1][lj][lk];
            double pB=ss[li][lj-1][lk]?pC:sx[li][lj-1][lk],pT=ss[li][lj+1][lk]?pC:sx[li][lj+1][lk];
            double pF=ss[li][lj][lk-1]?pC:sx[li][lj][lk-1],pK=ss[li][lj][lk+1]?pC:sx[li][lj][lk+1];
            double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
            double ed=diag;
            if(ss[li-1][lj][lk])ed-=idx2;if(ss[li+1][lj][lk])ed-=idx2;
            if(ss[li][lj-1][lk])ed-=idy2;if(ss[li][lj+1][lk])ed-=idy2;
            if(ss[li][lj][lk-1])ed-=idz2;if(ss[li][lj][lk+1])ed-=idz2;
            double inv=(ed==diag)?inv_diag:((ed<1e-15)?0.0:1.0/ed);
            sx[li][lj][lk]+=inv*(my_b-diag*pC+lap);
        }
        __syncthreads();
    }

    if(valid&&!ss[li][lj][lk])x[gid]=sx[li][lj][lk];
}

// ── Utility kernels ──
__global__ void restrict_opt_kernel(
    const double *x, const double *b, const bool *solid,
    double *b_coarse, bool *solid_coarse,
    int fnx, int fny, int fnz, int fpitch,
    double idx2, double idy2, double idz2, double diag,
    int cstride)
{
    int ic=blockIdx.x*blockDim.x+threadIdx.x+1,jc=blockIdx.y*blockDim.y+threadIdx.y+1,kc=blockIdx.z*blockDim.z+threadIdx.z+1;
    int cnx=fnx/2,cny=fny/2,cnz=fnz/2;
    if(ic>cnx||jc>cny||kc>cnz)return;
    int i_f=2*ic-1,j_f=2*jc-1,k_f=2*kc-1;double sum=0;int cnt=0;
    for(int di=0;di<2;di++)for(int dj=0;dj<2;dj++)for(int dk=0;dk<2;dk++){
        int fi=i_f+di,fj=j_f+dj,fk=k_f+dk;
        int fidx=opti_idx(fi,fj,fk,fpitch,fny);
        if(solid[fidx])continue;
        double pC=x[fidx];
        double pL=(fi>1&&!solid[opti_idx(fi-1,fj,fk,fpitch,fny)])?x[opti_idx(fi-1,fj,fk,fpitch,fny)]:pC;
        double pR=(fi<fnx&&!solid[opti_idx(fi+1,fj,fk,fpitch,fny)])?x[opti_idx(fi+1,fj,fk,fpitch,fny)]:pC;
        double pB=(fj>1&&!solid[opti_idx(fi,fj-1,fk,fpitch,fny)])?x[opti_idx(fi,fj-1,fk,fpitch,fny)]:pC;
        double pT=(fj<fny&&!solid[opti_idx(fi,fj+1,fk,fpitch,fny)])?x[opti_idx(fi,fj+1,fk,fpitch,fny)]:pC;
        double pF=(fk>1&&!solid[opti_idx(fi,fj,fk-1,fpitch,fny)])?x[opti_idx(fi,fj,fk-1,fpitch,fny)]:pC;
        double pK=(fk<fnz&&!solid[opti_idx(fi,fj,fk+1,fpitch,fny)])?x[opti_idx(fi,fj,fk+1,fpitch,fny)]:pC;
        double lap=(pL+pR)*idx2+(pB+pT)*idy2+(pF+pK)*idz2;
        sum+=b[fidx]-diag*pC+lap;cnt++;
    }
    int cid=opti_idx(ic,jc,kc,cstride,cny);
    if(!solid_coarse[cid]&&cnt>0)b_coarse[cid]=sum/cnt;
}

__global__ void prolong_opt_kernel(
    double *x_fine, const double *x_coarse, const bool *solid_fine,
    int fnx, int fny, int fnz, int fpitch, int cstride)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x+1,j=blockIdx.y*blockDim.y+threadIdx.y+1,k=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(i>fnx||j>fny||k>fnz)return;
    int fid=opti_idx(i,j,k,fpitch,fny);
    if(solid_fine[fid])return;
    int ic=(i+1)/2,jc=(j+1)/2,kc=(k+1)/2;
    x_fine[fid]+=2.0*x_coarse[opti_idx(ic,jc,kc,cstride,fny/2)];
}

__global__ void zero_kernel_opt(double *a, int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)a[i]=0.0;}
__global__ void copy_kernel_opt(double *dst, const double *src, int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)dst[i]=src[i];}
__global__ void restrict_solid_opt_kernel(
    const bool *sf, bool *sc, int fnx, int fny, int fnz, int fpitch,
    int cnx, int cny, int cnz, int cpitch)
{
    int ic=blockIdx.x*blockDim.x+threadIdx.x+1,jc=blockIdx.y*blockDim.y+threadIdx.y+1,kc=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ic>cnx||jc>cny||kc>cnz)return;
    int i_f=2*ic-1,j_f=2*jc-1,k_f=2*kc-1,scount=0;
    for(int di=0;di<2;di++)for(int dj=0;dj<2;dj++)for(int dk=0;dk<2;dk++)
        if(sf[opti_idx(i_f+di,j_f+dj,k_f+dk,fpitch,fny)])scount++;
    sc[opti_idx(ic,jc,kc,cpitch,cny)]=(scount>=4);
}

// ═══════════════════════════════════════════════════════════════
//  Optimized V-Cycle (paper Section 5.3, full aggregation)
//
//  V(nu, nu) cycle: nu pre-smooth + coarsen + nu post-smooth.
//    Down:  (nu-1) smooth-only + 1 aggregated (smooth + restrict)
//    Coarse: 1 kernel (20 sweeps fused)
//    Up:    1 aggregated (prolong + smooth) + (nu-1) smooth-only
//  Setting nu=2 (V(2,2)) typically halves PCG iteration count for Poisson
//  in exchange for ~1.5–1.7× the per-cycle cost. Default to nu=2.
// ═══════════════════════════════════════════════════════════════
static constexpr int VCYCLE_NU = 1;  // V(1,1) — matches paper Algorithm 1

static void vCycle_opt(CudaUAAMGPreconditioner3D::Level* levels, int lv, int nl, cudaStream_t stream) {
    auto& L = levels[lv];
    int nx=L.g.nx, ny=L.g.ny, nz=L.g.nz;
    dim3 block(TILED_T,TILED_T,TILED_T);
    dim3 grid((nx+TILED_T-1)/TILED_T,(ny+TILED_T-1)/TILED_T,(nz+TILED_T-1)/TILED_T);

    if(lv==nl-1){
        rbgs_coarsest_kernel<<<grid,block,0,stream>>>(L.g.x,L.g.b,L.g.solid,nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag);
        return;
    }

    auto& coarse=levels[lv+1];
    int cnx=coarse.g.nx,cny=coarse.g.ny,cnz=coarse.g.nz;
    int Nc=(cnx+2)*(cny+2)*(cnz+2);

    // Down: (nu-1) smooth-only passes + 1 fused smooth+restrict
    for(int s=0;s<VCYCLE_NU-1;s++){
        rbgs_smooth_only_kernel<<<grid,block,0,stream>>>(
            L.g.x,L.g.b,L.g.solid,nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag);
    }
    cudaMemsetAsync(coarse.g.b, 0, Nc*sizeof(double), stream);
    rbgs_restrict_aggregated_kernel<<<grid,block,0,stream>>>(
        L.g.x,L.g.b,L.g.solid,coarse.g.b,coarse.g.solid,
        nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag,
        cnx,cny,coarse.g.pitch);

    vCycle_opt(levels,lv+1,nl,stream);

    // Up: 1 fused prolong+smooth + (nu-1) smooth-only
    prolong_rbgs_aggregated_kernel<<<grid,block,0,stream>>>(
        L.g.x,coarse.g.x,L.g.b,L.g.solid,
        nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag,
        cny,coarse.g.pitch);
    for(int s=0;s<VCYCLE_NU-1;s++){
        rbgs_smooth_only_kernel<<<grid,block,0,stream>>>(
            L.g.x,L.g.b,L.g.solid,nx,ny,nz,L.g.pitch,L.g.idx2,L.g.idy2,L.g.idz2,L.g.diag);
    }
}

// Pure V-cycle (no setup) for accurate timing. Caller is responsible for
// pre-loading b at level 0 and zeroing x at all levels (which apply_optimized
// already does on its first call — so use vcycle_only between PCG iters
// to measure the actual V-cycle cost without setup overhead).
void CudaUAAMGPreconditioner3D::vcycle_only() {
    int nl = (int)levels_.size();
    if (nl == 0) return;
    vCycle_opt(levels_.data(), 0, nl, 0);  // no sync — caller times across many calls
}

void CudaUAAMGPreconditioner3D::apply_optimized(const CudaGrid3D& fine, const double* r, double* z) {
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
        restrict_solid_opt_kernel<<<grid,block,0,stream>>>(
            fL.g.solid,cL.g.solid,fL.g.nx,fL.g.ny,fL.g.nz,fL.g.pitch,
            cL.g.nx,cL.g.ny,cL.g.nz,cL.g.pitch);
    }

    copy_kernel_opt<<<(N0+255)/256,256,0,stream>>>(levels_[0].g.b,r,N0);
    for(int l=0;l<nl;l++){
        int N=(levels_[l].g.nx+2)*(levels_[l].g.ny+2)*(levels_[l].g.nz+2);
        cudaMemsetAsync(levels_[l].g.x, 0, N*sizeof(double), stream);
    }

    vCycle_opt(levels_.data(),0,nl,stream);
    cudaDeviceSynchronize();

    copy_kernel_opt<<<(N0+255)/256,256,0,stream>>>(z,levels_[0].g.x,N0);
    cudaDeviceSynchronize();
}


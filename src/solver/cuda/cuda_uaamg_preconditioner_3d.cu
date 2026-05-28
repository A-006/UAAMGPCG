/**
 * @file cuda_uaamg_preconditioner_3d.cu
 * @brief CUDA 3D UAAMG V-cycle preconditioner — matrix-free Galerkin, templated
 *        on the scalar type T (double / float).
 *
 * Paper-faithful (Sun et al. SIGGRAPH 2025, Algorithm 3):
 *   - constant prolongation with ×2 correction scaling (Eq. 11);
 *   - 8-to-1 restriction R = Pᵀ (sum over 2×2×2 children);
 *   - Galerkin coarse operator A_{l+1} = R_l A_l P_l (Eq. 12), matrix-free:
 *     coarse +x coupling = sum of the 4 fine +x couplings across the shared face;
 *   - RBGS smoother, symmetric V(1,1);
 *   - §5.4 coefficient trimming (uniform tiles use a default stencil).
 *
 * Instantiated for double and float at the bottom. FP32 halves memory traffic.
 */
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"
#include <cstdlib>

__device__ inline int dev_idx3d(int i, int j, int k, int pitch, int ny) {
    return i + j * pitch + k * pitch * (ny + 2);
}

// ── Finest-level stencil from the solid mask ──
template<typename T>
__global__ void setup_fine_coeffs_kernel_3d(
    const bool *solid, T *cx, T *cy, T *cz,
    int nx, int ny, int nz, int pitch, T idx2, T idy2, T idz2)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (i>nx || j>ny || k>nz) return;
    int id = dev_idx3d(i,j,k,pitch,ny);
    if (solid[id]) { cx[id]=cy[id]=cz[id]=T(0); return; }
    cx[id] = (i<nx && !solid[dev_idx3d(i+1,j,k,pitch,ny)]) ? idx2 : T(0);
    cy[id] = (j<ny && !solid[dev_idx3d(i,j+1,k,pitch,ny)]) ? idy2 : T(0);
    cz[id] = (k<nz && !solid[dev_idx3d(i,j,k+1,pitch,ny)]) ? idz2 : T(0);
}

// diag[c] = sum of the 6 active couplings (fine or coarse level)
template<typename T>
__global__ void setup_diag_kernel_3d(
    const bool *solid, const T *cx, const T *cy, const T *cz,
    T *diag, int nx, int ny, int nz, int pitch)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (i>nx || j>ny || k>nz) return;
    int id = dev_idx3d(i,j,k,pitch,ny);
    if (solid[id]) { diag[id]=T(0); return; }
    diag[id] = cx[id] + cx[dev_idx3d(i-1,j,k,pitch,ny)]
             + cy[id] + cy[dev_idx3d(i,j-1,k,pitch,ny)]
             + cz[id] + cz[dev_idx3d(i,j,k-1,pitch,ny)];
}

// ── Galerkin coarse couplings: sum the 4 fine couplings on each shared face ──
template<typename T>
__global__ void galerkin_coeffs_kernel_3d(
    const T *fcx, const T *fcy, const T *fcz,
    const bool *csolid, T *ccx, T *ccy, T *ccz,
    int fnx, int fny, int fpitch, int cnx, int cny, int cnz, int cpitch)
{
    int ic = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (ic>cnx || jc>cny || kc>cnz) return;
    int cid = dev_idx3d(ic,jc,kc,cpitch,cny);
    if (csolid[cid]) { ccx[cid]=ccy[cid]=ccz[cid]=T(0); return; }
    int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1;
    T sx=0, sy=0, sz=0;
    for (int dj=0; dj<2; dj++) for (int dk=0; dk<2; dk++)
        sx += fcx[dev_idx3d(i_f+1, j_f+dj, k_f+dk, fpitch, fny)];   // +x face = fine i_f+1
    for (int di=0; di<2; di++) for (int dk=0; dk<2; dk++)
        sy += fcy[dev_idx3d(i_f+di, j_f+1, k_f+dk, fpitch, fny)];
    for (int di=0; di<2; di++) for (int dj=0; dj<2; dj++)
        sz += fcz[dev_idx3d(i_f+di, j_f+dj, k_f+1, fpitch, fny)];
    ccx[cid]=sx; ccy[cid]=sy; ccz[cid]=sz;
}

// ── §5.4 mark trimmed tiles (tile + 1-ring all uniform-default) ──
template<typename T>
__global__ void mark_trimmed_kernel_3d(
    const T *diag, T diagd, int nx, int ny, int nz, int pitch,
    int ntx, int nty, int ntz, bool *trimmed)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= ntx*nty*ntz) return;
    int bx=t%ntx, by=(t/ntx)%nty, bz=t/(ntx*nty);
    bool trim=true;
    for(int gi=8*bx; gi<=8*bx+9 && trim; gi++)
      for(int gj=8*by; gj<=8*by+9 && trim; gj++)
        for(int gk=8*bz; gk<=8*bz+9 && trim; gk++){
            if(gi<1||gi>nx||gj<1||gj>ny||gk<1||gk>nz){ trim=false; break; }
            T d=diag[dev_idx3d(gi,gj,gk,pitch,ny)];
            if(fabs(double(d-diagd)) > 1e-6*double(diagd)){ trim=false; break; }
        }
    trimmed[t]=trim;
}

// ── RBGS sweep (one parity), stored coeffs + §5.4 trimming ──
template<typename T>
__global__ void rbgs_coeff_kernel_3d(
    T *x, const T *b, const bool *solid,
    const T *diag, const T *cx, const T *cy, const T *cz,
    int nx, int ny, int nz, int pitch, int parity,
    const bool *trimmed, T cxd, T cyd, T czd, T diagd)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (i>nx || j>ny || k>nz) return;
    if (((i+j+k)&1) != parity) return;
    int id = dev_idx3d(i,j,k,pitch,ny);
    if (solid[id]) return;
    int ip=dev_idx3d(i+1,j,k,pitch,ny), jp=dev_idx3d(i,j+1,k,pitch,ny), kp=dev_idx3d(i,j,k+1,pitch,ny);
    int im=dev_idx3d(i-1,j,k,pitch,ny), jm=dev_idx3d(i,j-1,k,pitch,ny), km=dev_idx3d(i,j,k-1,pitch,ny);
    int tileid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    T Cxp,Cxm,Cyp,Cym,Czp,Czm,D;
    if (trimmed[tileid]) {                          // uniform tile → default stencil
        Cxp=Cxm=cxd; Cyp=Cym=cyd; Czp=Czm=czd; D=diagd;
    } else {
        D=diag[id]; if (D < T(1e-30)) return;
        Cxp=cx[id]; Cxm=cx[im]; Cyp=cy[id]; Cym=cy[jm]; Czp=cz[id]; Czm=cz[km];
    }
    T nb = Cxp*x[ip] + Cxm*x[im] + Cyp*x[jp] + Cym*x[jm] + Czp*x[kp] + Czm*x[km];
    x[id] = (b[id] + nb) / D;
}

// ── Residual restriction R = Pᵀ (sum over 2×2×2 children) ──
template<typename T>
__global__ void restrict_coeff_kernel_3d(
    const T *xf, const T *bf, const bool *fsolid,
    const T *fdiag, const T *fcx, const T *fcy, const T *fcz,
    T *bc, const bool *csolid,
    int fnx, int fny, int fnz, int fpitch, int cnx, int cny, int cnz, int cpitch)
{
    int ic = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (ic>cnx || jc>cny || kc>cnz) return;
    int cid = dev_idx3d(ic,jc,kc,cpitch,cny);
    if (csolid[cid]) return;
    int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1;
    T sum=0;
    for (int di=0;di<2;di++) for(int dj=0;dj<2;dj++) for(int dk=0;dk<2;dk++) {
        int fi=i_f+di, fj=j_f+dj, fk=k_f+dk, fid=dev_idx3d(fi,fj,fk,fpitch,fny);
        if (fsolid[fid]) continue;
        int im=dev_idx3d(fi-1,fj,fk,fpitch,fny), jm=dev_idx3d(fi,fj-1,fk,fpitch,fny), km=dev_idx3d(fi,fj,fk-1,fpitch,fny);
        T Ax = fdiag[fid]*xf[fid]
             - fcx[fid]*xf[dev_idx3d(fi+1,fj,fk,fpitch,fny)] - fcx[im]*xf[im]
             - fcy[fid]*xf[dev_idx3d(fi,fj+1,fk,fpitch,fny)] - fcy[jm]*xf[jm]
             - fcz[fid]*xf[dev_idx3d(fi,fj,fk+1,fpitch,fny)] - fcz[km]*xf[km];
        sum += bf[fid] - Ax;
    }
    bc[cid] = sum;
}

// ── Prolongation: constant injection with ×2 scaling (paper Eq. 11) ──
template<typename T>
__global__ void prolong_kernel_3d(
    T *x_fine, const T *x_coarse, const bool *solid_fine,
    int fnx, int fny, int fnz, int fpitch, int cpitch)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (i>fnx || j>fny || k>fnz) return;
    int fid = dev_idx3d(i,j,k,fpitch,fny);
    if (solid_fine[fid]) return;
    int ic=(i+1)/2, jc=(j+1)/2, kc=(k+1)/2, cny=fny/2;
    x_fine[fid] += T(2) * x_coarse[dev_idx3d(ic,jc,kc,cpitch,cny)];
}

// ── Restrict solid mask (no scalar T) ──
__global__ void restrict_solid_kernel_3d(
    const bool *solid_fine, bool *solid_coarse,
    int fnx, int fny, int fnz, int fpitch, int cpitch)
{
    int ic = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int kc = blockIdx.z*blockDim.z + threadIdx.z + 1;
    int cnx = fnx/2, cny = fny/2, cnz = fnz/2;
    if (ic>cnx || jc>cny || kc>cnz) return;
    int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1, sc=0;
    for (int di=0;di<2;di++) for (int dj=0;dj<2;dj++) for (int dk=0;dk<2;dk++)
        if (solid_fine[dev_idx3d(i_f+di,j_f+dj,k_f+dk,fpitch,fny)]) sc++;
    solid_coarse[dev_idx3d(ic,jc,kc,cpitch,cny)] = (sc >= 4);
}

template<typename T>
__global__ void zero_kernel_3d(T *a, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) a[i] = T(0);
}
template<typename T>
__global__ void copy_kernel_3d(T *dst, const T *src, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ── Shared-memory tiled RBGS pass (one parity) ──
// Loads the 8³ tile + 1-cell face halo of x into shared once; the active-parity
// cells read their 6 neighbours from shared instead of 6 strided global reads.
// EXACT (not block-RBGS): each colour pass writes back to global, so the next
// pass loads an up-to-date halo — convergence is identical to the untiled sweep.
// Reduces the dominant global x traffic → helps the memory-bound FP32 regime
// (ncu: FP32 V-cycle is 71% DRAM-bound; FP64 is compute-bound so this is neutral).
template<typename T>
__global__ void rbgs_tiled_pass_kernel_3d(
    T* x, const T* b, const bool* solid,
    const T* diag, const T* cx, const T* cy, const T* cz,
    int nx,int ny,int nz,int pitch, int parity,
    const bool* trimmed, T cxd, T cyd, T czd, T diagd)
{
    __shared__ T sx[10][10][10];
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int gi=blockIdx.x*8+tx+1, gj=blockIdx.y*8+ty+1, gk=blockIdx.z*8+tz+1;
    int li=tx+1, lj=ty+1, lk=tz+1;
    bool valid=(gi<=nx&&gj<=ny&&gk<=nz);
    sx[li][lj][lk] = valid ? x[dev_idx3d(gi,gj,gk,pitch,ny)] : T(0);
    if(tx==0)  sx[0][lj][lk]  = (gi>1 &&gj<=ny&&gk<=nz)? x[dev_idx3d(gi-1,gj,gk,pitch,ny)] : T(0);
    if(tx==7)  sx[9][lj][lk]  = (gi<nx&&gj<=ny&&gk<=nz)? x[dev_idx3d(gi+1,gj,gk,pitch,ny)] : T(0);
    if(ty==0)  sx[li][0][lk]  = (gi<=nx&&gj>1 &&gk<=nz)? x[dev_idx3d(gi,gj-1,gk,pitch,ny)] : T(0);
    if(ty==7)  sx[li][9][lk]  = (gi<=nx&&gj<ny&&gk<=nz)? x[dev_idx3d(gi,gj+1,gk,pitch,ny)] : T(0);
    if(tz==0)  sx[li][lj][0]  = (gi<=nx&&gj<=ny&&gk>1 )? x[dev_idx3d(gi,gj,gk-1,pitch,ny)] : T(0);
    if(tz==7)  sx[li][lj][9]  = (gi<=nx&&gj<=ny&&gk<nz)? x[dev_idx3d(gi,gj,gk+1,pitch,ny)] : T(0);
    __syncthreads();
    if(!valid || ((gi+gj+gk)&1)!=parity) return;
    int gid=dev_idx3d(gi,gj,gk,pitch,ny);
    if(solid[gid]) return;
    int tileid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    T cxp,cxm,cyp,cym,czp,czm,D;
    if(trimmed[tileid]){ cxp=cxm=cxd; cyp=cym=cyd; czp=czm=czd; D=diagd; }
    else {
        D=diag[gid]; if(D<T(1e-30)) return;
        int im=dev_idx3d(gi-1,gj,gk,pitch,ny), jm=dev_idx3d(gi,gj-1,gk,pitch,ny), km=dev_idx3d(gi,gj,gk-1,pitch,ny);
        cxp=cx[gid]; cxm=cx[im]; cyp=cy[gid]; cym=cy[jm]; czp=cz[gid]; czm=cz[km];
    }
    T nb = cxp*sx[li+1][lj][lk] + cxm*sx[li-1][lj][lk]
         + cyp*sx[li][lj+1][lk] + cym*sx[li][lj-1][lk]
         + czp*sx[li][lj][lk+1] + czm*sx[li][lj][lk-1];
    x[gid] = (b[gid] + nb) / D;
}

// ── one RBGS sweep (forward = odd,even ; reverse = even,odd) ──
template<typename T>
static void rbgs_sweep_3d(typename CudaUAAMGPreconditioner3DT<T>::Level& L, cudaStream_t stream, bool reverse) {
    int nx=L.g.nx, ny=L.g.ny, nz=L.g.nz;
    dim3 block(8,8,8), grid((nx+7)/8,(ny+7)/8,(nz+7)/8);
    int p0 = reverse ? 0 : 1;
    // Shared-mem tiling pays off only on large levels (it adds load/occupancy
    // overhead that dominates on small/coarse levels). Gate by level size so
    // tiling is a pure win: tiled on the fine levels, plain on the coarse ones.
    bool tiled = ((long)nx*ny*nz >= (1L<<21));   // ≥ ~2M cells
    auto launch=[&](int par){
        if(tiled)
            rbgs_tiled_pass_kernel_3d<T><<<grid,block,0,stream>>>(L.g.x,L.g.b,L.g.solid,L.diag,L.cx,L.cy,L.cz,nx,ny,nz,L.g.pitch,par,
                L.trimmed,L.cxd,L.cyd,L.czd,L.diagd);
        else
            rbgs_coeff_kernel_3d<T><<<grid,block,0,stream>>>(L.g.x,L.g.b,L.g.solid,L.diag,L.cx,L.cy,L.cz,nx,ny,nz,L.g.pitch,par,
                L.trimmed,L.cxd,L.cyd,L.czd,L.diagd);
    };
    launch(p0); launch(1-p0);
}

// ── Symmetric V(1,1) ──
template<typename T>
static void vCycle3D(typename CudaUAAMGPreconditioner3DT<T>::Level* levels, int level, int nlevels, cudaStream_t stream) {
    auto& L = levels[level];
    int nx=L.g.nx, ny=L.g.ny, nz=L.g.nz;
    if (level == nlevels-1) {
        for (int s=0;s<10;s++) { rbgs_sweep_3d<T>(L,stream,false); rbgs_sweep_3d<T>(L,stream,true); }
        return;
    }
    rbgs_sweep_3d<T>(L, stream, false);                     // pre-smooth (forward)

    auto& coarse = levels[level+1];
    int cnx=coarse.g.nx, cny=coarse.g.ny, cnz=coarse.g.nz;
    dim3 cblock(8,8,8), cgrid((cnx+7)/8,(cny+7)/8,(cnz+7)/8);
    restrict_coeff_kernel_3d<T><<<cgrid,cblock,0,stream>>>(
        L.g.x, L.g.b, L.g.solid, L.diag, L.cx, L.cy, L.cz,
        coarse.g.b, coarse.g.solid,
        nx, ny, nz, L.g.pitch, cnx, cny, cnz, coarse.g.pitch);

    int Nc=(cnx+2)*(cny+2)*(cnz+2);
    zero_kernel_3d<T><<<(Nc+255)/256,256,0,stream>>>(coarse.g.x, Nc);
    vCycle3D<T>(levels, level+1, nlevels, stream);

    dim3 fblock(8,8,8), fgrid((nx+7)/8,(ny+7)/8,(nz+7)/8);
    prolong_kernel_3d<T><<<fgrid,fblock,0,stream>>>(
        L.g.x, coarse.g.x, L.g.solid, nx, ny, nz, L.g.pitch, coarse.g.pitch);

    rbgs_sweep_3d<T>(L, stream, true);                      // post-smooth (reverse → symmetric)
}

// ── CudaUAAMGPreconditioner3DT<T> ──

template<typename T>
void CudaUAAMGPreconditioner3DT<T>::build(const CudaGrid3DT_<T>& fine) {
    if (cached_nx_ == fine.nx && cached_ny_ == fine.ny && cached_nz_ == fine.nz) return;
    destroy();
    int nx = fine.nx, ny = fine.ny, nz = fine.nz;
    T dx = fine.dx, dy = fine.dy, dz = fine.dz;
    while (nx >= 2 && ny >= 2 && nz >= 2) {
        Level L;
        L.g.allocate(nx, ny, nz, dx, dy, dz);
        L.stride = nx + 2;
        int N = (nx+2)*(ny+2)*(nz+2);
        cudaMalloc(&L.diag, N*sizeof(T)); cudaMemset(L.diag,0,N*sizeof(T));
        cudaMalloc(&L.cx,   N*sizeof(T)); cudaMemset(L.cx,  0,N*sizeof(T));
        cudaMalloc(&L.cy,   N*sizeof(T)); cudaMemset(L.cy,  0,N*sizeof(T));
        cudaMalloc(&L.cz,   N*sizeof(T)); cudaMemset(L.cz,  0,N*sizeof(T));
        L.ntx=(nx+7)/8; L.nty=(ny+7)/8; L.ntz=(nz+7)/8;
        cudaMalloc(&L.trimmed, (size_t)L.ntx*L.nty*L.ntz*sizeof(bool));
        cudaMemset(L.trimmed, 0, (size_t)L.ntx*L.nty*L.ntz*sizeof(bool));
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4) break;
        nx /= 2; ny /= 2; nz /= 2; dx *= T(2); dy *= T(2); dz *= T(2);
    }
    cached_nx_ = fine.nx; cached_ny_ = fine.ny; cached_nz_ = fine.nz;
}

template<typename T>
void CudaUAAMGPreconditioner3DT<T>::setupLevels(const CudaGrid3DT_<T>& fine) {
    build(fine);
    int nl = (int)levels_.size();
    cudaStream_t stream = 0;
    int N0 = (fine.nx+2)*(fine.ny+2)*(fine.nz+2);

    cudaMemcpy(levels_[0].g.solid, fine.solid, N0 * sizeof(bool), cudaMemcpyDeviceToDevice);
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l-1]; auto& cL = levels_[l];
        dim3 block(8,8,8), grid((cL.g.nx+7)/8,(cL.g.ny+7)/8,(cL.g.nz+7)/8);
        restrict_solid_kernel_3d<<<grid,block,0,stream>>>(
            fL.g.solid, cL.g.solid, fL.g.nx, fL.g.ny, fL.g.nz, fL.g.pitch, cL.g.pitch);
    }
    {
        auto& L = levels_[0]; int nx=L.g.nx, ny=L.g.ny, nz=L.g.nz;
        dim3 block(8,8,8), grid((nx+7)/8,(ny+7)/8,(nz+7)/8);
        setup_fine_coeffs_kernel_3d<T><<<grid,block,0,stream>>>(
            L.g.solid, L.cx, L.cy, L.cz, nx, ny, nz, L.g.pitch, L.g.idx2, L.g.idy2, L.g.idz2);
        setup_diag_kernel_3d<T><<<grid,block,0,stream>>>(
            L.g.solid, L.cx, L.cy, L.cz, L.diag, nx, ny, nz, L.g.pitch);
    }
    for (int l = 1; l < nl; l++) {
        auto& fL = levels_[l-1]; auto& cL = levels_[l];
        int cnx=cL.g.nx, cny=cL.g.ny, cnz=cL.g.nz;
        dim3 block(8,8,8), grid((cnx+7)/8,(cny+7)/8,(cnz+7)/8);
        galerkin_coeffs_kernel_3d<T><<<grid,block,0,stream>>>(
            fL.cx, fL.cy, fL.cz, cL.g.solid, cL.cx, cL.cy, cL.cz,
            fL.g.nx, fL.g.ny, fL.g.pitch, cnx, cny, cnz, cL.g.pitch);
        setup_diag_kernel_3d<T><<<grid,block,0,stream>>>(
            cL.g.solid, cL.cx, cL.cy, cL.cz, cL.diag, cnx, cny, cnz, cL.g.pitch);
    }

    // §5.4: per-level uniform default stencil (Galerkin: coarse coupling = 4× finer)
    levels_[0].cxd = levels_[0].g.idx2; levels_[0].cyd = levels_[0].g.idy2; levels_[0].czd = levels_[0].g.idz2;
    for (int l = 1; l < nl; l++) {
        levels_[l].cxd = T(4)*levels_[l-1].cxd;
        levels_[l].cyd = T(4)*levels_[l-1].cyd;
        levels_[l].czd = T(4)*levels_[l-1].czd;
    }
    static bool notrim = (std::getenv("UAAMG_NOTRIM") != nullptr);
    for (int l = 0; l < nl; l++) {
        auto& L = levels_[l];
        L.diagd = L.cxd+L.cxd + L.cyd+L.cyd + L.czd+L.czd;   // match setup_diag summation
        int ntiles = L.ntx*L.nty*L.ntz;
        if (notrim) {
            cudaMemsetAsync(L.trimmed, 0, ntiles*sizeof(bool), stream);
        } else {
            mark_trimmed_kernel_3d<T><<<(ntiles+255)/256,256,0,stream>>>(
                L.diag, L.diagd, L.g.nx, L.g.ny, L.g.nz, L.g.pitch, L.ntx, L.nty, L.ntz, L.trimmed);
        }
    }
}

template<typename T>
void CudaUAAMGPreconditioner3DT<T>::vcycle_apply(const CudaGrid3DT_<T>& fine, const T* r, T* z) {
    int nl = (int)levels_.size();
    cudaStream_t stream = 0;
    int N0 = (fine.nx+2)*(fine.ny+2)*(fine.nz+2);
    copy_kernel_3d<T><<<(N0+255)/256,256,0,stream>>>(levels_[0].g.b, r, N0);
    for (int l = 0; l < nl; l++) {
        int N=(levels_[l].g.nx+2)*(levels_[l].g.ny+2)*(levels_[l].g.nz+2);
        zero_kernel_3d<T><<<(N+255)/256,256,0,stream>>>(levels_[l].g.x, N);
    }
    vCycle3D<T>(levels_.data(), 0, nl, stream);
    cudaDeviceSynchronize();
    copy_kernel_3d<T><<<(N0+255)/256,256,0,stream>>>(z, levels_[0].g.x, N0);
    cudaDeviceSynchronize();
}

template<typename T>
void CudaUAAMGPreconditioner3DT<T>::apply(const CudaGrid3DT_<T>& fine, const T* r, T* z) {
    setupLevels(fine);
    vcycle_apply(fine, r, z);
}

template<typename T>
void CudaUAAMGPreconditioner3DT<T>::destroy() {
    for (auto& L : levels_) {
        L.g.free();
        if (L.diag) cudaFree(L.diag);
        if (L.cx)   cudaFree(L.cx);
        if (L.cy)   cudaFree(L.cy);
        if (L.cz)   cudaFree(L.cz);
        if (L.trimmed) cudaFree(L.trimmed);
        L.diag=L.cx=L.cy=L.cz=nullptr; L.trimmed=nullptr;
    }
    levels_.clear();
    cached_nx_ = cached_ny_ = cached_nz_ = -1;
}

// Explicit instantiations
template class CudaUAAMGPreconditioner3DT<double>;
template class CudaUAAMGPreconditioner3DT<float>;

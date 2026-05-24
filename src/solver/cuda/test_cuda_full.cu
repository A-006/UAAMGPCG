/**
 * @file test_cuda_full.cu
 * @brief Comprehensive GPU vs CPU tests — every kernel + full PCG tracing.
 *
 * Strategy:
 *   1. Test each GPU kernel independently (bit-exact match to CPU)
 *   2. Test one full PCG iteration (all kernels chained)
 *   3. Trace PCG iteration-by-iteration to find divergence point
 */
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; }} while(0)

// ── GPU kernels (exact copies from cuda_uaamg_preconditioner.cu + cuda_pcg.cu) ──

__device__ inline int dev_idx(int i, int j, int stride) { return i + j * stride; }

// RBGS pass1: odd-sum (matches CPU first checkerboard pass)
__global__ void krbgs1(double *x, const double *b, const bool *s,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    if (!((i + j) & 1)) return;
    int id = dev_idx(i, j, stride); if (s[id]) return;
    double pC = x[id];
    double pL = (i>1 && !s[dev_idx(i-1,j,stride)]) ? x[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !s[dev_idx(i+1,j,stride)]) ? x[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !s[dev_idx(i,j-1,stride)]) ? x[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !s[dev_idx(i,j+1,stride)]) ? x[dev_idx(i,j+1,stride)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
    double eff_d = diag;
    if (i==1||s[dev_idx(i-1,j,stride)]) eff_d -= idx2;
    if (i==nx||s[dev_idx(i+1,j,stride)]) eff_d -= idx2;
    if (j==1||s[dev_idx(i,j-1,stride)]) eff_d -= idy2;
    if (j==ny||s[dev_idx(i,j+1,stride)]) eff_d -= idy2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

__global__ void krbgs2(double *x, const double *b, const bool *s,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    if ((i + j) & 1) return;
    int id = dev_idx(i, j, stride); if (s[id]) return;
    double pC = x[id];
    double pL = (i>1 && !s[dev_idx(i-1,j,stride)]) ? x[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !s[dev_idx(i+1,j,stride)]) ? x[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !s[dev_idx(i,j-1,stride)]) ? x[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !s[dev_idx(i,j+1,stride)]) ? x[dev_idx(i,j+1,stride)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
    double eff_d = diag;
    if (i==1||s[dev_idx(i-1,j,stride)]) eff_d -= idx2;
    if (i==nx||s[dev_idx(i+1,j,stride)]) eff_d -= idx2;
    if (j==1||s[dev_idx(i,j-1,stride)]) eff_d -= idy2;
    if (j==ny||s[dev_idx(i,j+1,stride)]) eff_d -= idy2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

__global__ void kmatvec(const double *p, double *Ap, const bool *s,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int id = dev_idx(i, j, stride);
    if (s[id]) { Ap[id] = 0.0; return; }
    double pC = p[id];
    double pL = (i>1 && !s[dev_idx(i-1,j,stride)]) ? p[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !s[dev_idx(i+1,j,stride)]) ? p[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !s[dev_idx(i,j-1,stride)]) ? p[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !s[dev_idx(i,j+1,stride)]) ? p[dev_idx(i,j+1,stride)] : pC;
    Ap[id] = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
}

__global__ void krestrict(const double *xf, const double *bf, const bool *sf,
    double *bc, bool *sc, int fnx, int fny, int fs, int cs,
    double idx2, double idy2, double diag)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jc = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int cnx = fnx/2, cny = fny/2;
    if (ic > cnx || jc > cny) return;
    int i_f = 2*ic-1, j_f = 2*jc-1;
    double sum = 0; int cnt = 0;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++) {
            int fi = i_f+di, fj = j_f+dj, fid = dev_idx(fi,fj,fs);
            if (sf[fid]) continue;
            double pC = xf[fid];
            double pL = (fi>1 && !sf[dev_idx(fi-1,fj,fs)]) ? xf[dev_idx(fi-1,fj,fs)] : pC;
            double pR = (fi<fnx && !sf[dev_idx(fi+1,fj,fs)]) ? xf[dev_idx(fi+1,fj,fs)] : pC;
            double pB = (fj>1 && !sf[dev_idx(fi,fj-1,fs)]) ? xf[dev_idx(fi,fj-1,fs)] : pC;
            double pT = (fj<fny && !sf[dev_idx(fi,fj+1,fs)]) ? xf[dev_idx(fi,fj+1,fs)] : pC;
            double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
            sum += bf[fid] - diag * pC + lap;
            cnt++;
        }
    int cid = dev_idx(ic, jc, cs);
    if (!sc[cid] && cnt > 0) bc[cid] = sum / cnt;
}

__global__ void kprolong(double *xf, const double *xc, const bool *sf,
    int fnx, int fny, int fs, int cs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > fnx || j > fny) return;
    int fid = dev_idx(i, j, fs);
    if (sf[fid]) return;
    int ic = (i+1)/2, jc = (j+1)/2;
    xf[fid] += 2.0 * xc[dev_idx(ic, jc, cs)];
}

__global__ void kdot(const double *a, const double *b, const bool *s, int N, double *part)
{
    __shared__ double sd[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x)
        if (!s[k]) sum += a[k] * b[k];
    sd[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) sd[tid] += sd[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sd[0];
}

__global__ void kaxpy(double *y, const double *x, double a, const bool *s, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !s[k]) y[k] += a * x[k];
}

__global__ void ksum(const double *v, const bool *s, int N, double *part)
{
    __shared__ double sd[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x)
        if (!s[k]) sum += v[k];
    sd[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) sd[tid] += sd[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sd[0];
}

__global__ void kcount(const bool *s, int N, int *part)
{
    __shared__ int sd[256];
    int tid = threadIdx.x;
    int sum = 0;
    for (int k = blockIdx.x * blockDim.x + tid; k < N; k += blockDim.x * gridDim.x)
        if (!s[k]) sum++;
    sd[tid] = sum; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) sd[tid] += sd[tid + st];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sd[0];
}

__global__ void ksubmean(double *v, double m, const bool *s, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N && !s[k]) v[k] -= m;
}

__global__ void kcopy(double *dst, const double *src, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) dst[k] = src[k];
}

// ── CPU reference functions ──

static void cpu_rbgs_pass1(double *x, const double *b, const bool *s, int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    for (int i = 1; i <= nx; i++)
        for (int j = 1 + (i%2); j <= ny; j += 2) {
            int id = i + j*stride; if (s[id]) continue;
            double pC = x[id];
            double pL = (i>1 && !s[(i-1)+j*stride]) ? x[(i-1)+j*stride] : pC;
            double pR = (i<nx && !s[(i+1)+j*stride]) ? x[(i+1)+j*stride] : pC;
            double pB = (j>1 && !s[i+(j-1)*stride]) ? x[i+(j-1)*stride] : pC;
            double pT = (j<ny && !s[i+(j+1)*stride]) ? x[i+(j+1)*stride] : pC;
            double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
            double eff_d = diag;
            if (i==1||s[(i-1)+j*stride]) eff_d -= idx2;
            if (i==nx||s[(i+1)+j*stride]) eff_d -= idx2;
            if (j==1||s[i+(j-1)*stride]) eff_d -= idy2;
            if (j==ny||s[i+(j+1)*stride]) eff_d -= idy2;
            x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
        }
}

static void cpu_rbgs_pass2(double *x, const double *b, const bool *s, int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    for (int i = 1; i <= nx; i++)
        for (int j = 1 + ((i+1)%2); j <= ny; j += 2) {
            int id = i + j*stride; if (s[id]) continue;
            double pC = x[id];
            double pL = (i>1 && !s[(i-1)+j*stride]) ? x[(i-1)+j*stride] : pC;
            double pR = (i<nx && !s[(i+1)+j*stride]) ? x[(i+1)+j*stride] : pC;
            double pB = (j>1 && !s[i+(j-1)*stride]) ? x[i+(j-1)*stride] : pC;
            double pT = (j<ny && !s[i+(j+1)*stride]) ? x[i+(j+1)*stride] : pC;
            double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
            double eff_d = diag;
            if (i==1||s[(i-1)+j*stride]) eff_d -= idx2;
            if (i==nx||s[(i+1)+j*stride]) eff_d -= idx2;
            if (j==1||s[i+(j-1)*stride]) eff_d -= idy2;
            if (j==ny||s[i+(j+1)*stride]) eff_d -= idy2;
            x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
        }
}

static void cpu_rbgs_full(double *x, const double *b, const bool *s, int nx, int ny, int stride, double idx2, double idy2, double diag)
{ cpu_rbgs_pass1(x,b,s,nx,ny,stride,idx2,idy2,diag); cpu_rbgs_pass2(x,b,s,nx,ny,stride,idx2,idy2,diag); }

static void cpu_matvec(const double *p, double *Ap, const bool *s, int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            int id = i + j*stride;
            if (s[id]) { Ap[id] = 0.0; continue; }
            double pC = p[id];
            double pL = (i>1 && !s[(i-1)+j*stride]) ? p[(i-1)+j*stride] : pC;
            double pR = (i<nx && !s[(i+1)+j*stride]) ? p[(i+1)+j*stride] : pC;
            double pB = (j>1 && !s[i+(j-1)*stride]) ? p[i+(j-1)*stride] : pC;
            double pT = (j<ny && !s[i+(j+1)*stride]) ? p[i+(j+1)*stride] : pC;
            Ap[id] = diag * pC - (pL+pR)*idx2 - (pB+pT)*idy2;
        }
}

static double cpu_dot(const double *a, const double *b, const bool *s, int N)
{ double t=0; for(int k=0;k<N;k++) if(!s[k]) t+=a[k]*b[k]; return t; }

static void cpu_axpy(double *y, const double *x, double a, const bool *s, int N)
{ for(int k=0;k<N;k++) if(!s[k]) y[k]+=a*x[k]; }

static double cpu_mean(const double *v, const bool *s, int N)
{ double t=0; int c=0; for(int k=0;k<N;k++) if(!s[k]){t+=v[k];c++;} return c>0?t/c:0; }

// ── GPU host helpers ──

static double gpu_dot_reduce(double *d_part, int nblocks)
{ std::vector<double> h(nblocks); cudaMemcpy(h.data(),d_part,nblocks*sizeof(double),cudaMemcpyDeviceToHost); double t=0; for(auto v:h)t+=v; return t; }
static int gpu_count_reduce(int *d_part, int nblocks)
{ std::vector<int> h(nblocks); cudaMemcpy(h.data(),d_part,nblocks*sizeof(int),cudaMemcpyDeviceToHost); int t=0; for(auto v:h)t+=v; return t; }

// ── Comparison helpers ──
static double max_diff(const double *a, const double *b, int N, const bool *skip=nullptr)
{ double m=0; for(int k=0;k<N;k++) if(!skip||!skip[k]) m=std::max(m,std::abs(a[k]-b[k])); return m; }
static int diff_count(const double *a, const double *b, int N, double tol, const bool *skip=nullptr)
{ int c=0; for(int k=0;k<N;k++) if((!skip||!skip[k])&&std::abs(a[k]-b[k])>tol) c++; return c; }

// ── Setup ──
struct TestCtx {
    int nx=16, ny=8, stride, N;
    double dx=2.0, dy=0.5;  // anisotropic → more interesting
    double idx2, idy2, diag;
    std::vector<double> x, b;
    std::vector<char> solid;

    void init() {
        stride=nx+2; N=(nx+2)*(ny+2);
        idx2=1.0/(dx*dx); idy2=1.0/(dy*dy); diag=2.0*(idx2+idy2);
        x.assign(N,0); b.assign(N,0); solid.assign(N,0);
        for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) b[i+j*stride]=std::sin(i*0.5)*std::cos(j*0.7);
        // Mark a few cells as solid
        for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) if(i==3&&j>=2&&j<=5) solid[i+j*stride]=1;
    }

    void to_gpu(double*& dx, double*& db, bool*& ds) {
        cudaMalloc(&dx,N*sizeof(double)); cudaMemcpy(dx,x.data(),N*sizeof(double),cudaMemcpyHostToDevice);
        cudaMalloc(&db,N*sizeof(double)); cudaMemcpy(db,b.data(),N*sizeof(double),cudaMemcpyHostToDevice);
        cudaMalloc(&ds,N*sizeof(bool));   cudaMemcpy(ds,solid.data(),N*sizeof(bool),cudaMemcpyHostToDevice);
    }
};

// ═══════════════════════════════════════════════════════════
//  Test 1: RBGS pass1 only
// ═══════════════════════════════════════════════════════════
static int test_rbgs_pass1(TestCtx& ctx) {
    printf("Test 1: RBGS pass1 (odd-sum)\n");
    std::vector<double> cpu = ctx.x;
    cpu_rbgs_pass1(cpu.data(),ctx.b.data(),(bool*)ctx.solid.data(),ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);

    double *dx,*db; bool *ds; ctx.to_gpu(dx,db,ds);
    dim3 b(16,16), g((ctx.nx+15)/16,(ctx.ny+15)/16);
    krbgs1<<<g,b>>>(dx,db,ds,ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    std::vector<double> gpu(ctx.N); cudaMemcpy(gpu.data(),dx,ctx.N*sizeof(double),cudaMemcpyDeviceToHost);
    double d=max_diff(cpu.data(),gpu.data(),ctx.N,(bool*)ctx.solid.data());
    int nd=diff_count(cpu.data(),gpu.data(),ctx.N,1e-14,(bool*)ctx.solid.data());
    printf("  max|cpu-gpu|=%e  cells diff>1e-14=%d/%d  %s\n",d,nd,ctx.nx*ctx.ny,d<1e-14?"PASS":"FAIL");
    cudaFree(dx);cudaFree(db);cudaFree(ds); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 2: RBGS pass2 only (after pass1 — tests combined)
// ═══════════════════════════════════════════════════════════
static int test_rbgs_pass2(TestCtx& ctx) {
    printf("Test 2: RBGS pass2 (after pass1)\n");
    std::vector<double> cpu = ctx.x;
    cpu_rbgs_pass1(cpu.data(),ctx.b.data(),(bool*)ctx.solid.data(),ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);
    cpu_rbgs_pass2(cpu.data(),ctx.b.data(),(bool*)ctx.solid.data(),ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);

    double *dx,*db; bool *ds; ctx.to_gpu(dx,db,ds);
    dim3 b(16,16), g((ctx.nx+15)/16,(ctx.ny+15)/16);
    krbgs1<<<g,b>>>(dx,db,ds,ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    krbgs2<<<g,b>>>(dx,db,ds,ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    std::vector<double> gpu(ctx.N); cudaMemcpy(gpu.data(),dx,ctx.N*sizeof(double),cudaMemcpyDeviceToHost);
    double d=max_diff(cpu.data(),gpu.data(),ctx.N,(bool*)ctx.solid.data());
    printf("  max|cpu-gpu|=%e  %s\n",d,d<1e-14?"PASS":"FAIL");
    cudaFree(dx);cudaFree(db);cudaFree(ds); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 3: Matvec
// ═══════════════════════════════════════════════════════════
static int test_matvec(TestCtx& ctx) {
    printf("Test 3: Matvec\n");
    std::vector<double> p(ctx.N); for(int k=0;k<ctx.N;k++) p[k]=std::sin(k*0.1);
    std::vector<double> cpu(ctx.N);
    cpu_matvec(p.data(),cpu.data(),(bool*)ctx.solid.data(),ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);

    double *dp,*dAp; bool *ds;
    CUDA_CHECK(cudaMalloc(&dp,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(dp,p.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dAp,ctx.N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ds,ctx.N*sizeof(bool)));   CUDA_CHECK(cudaMemcpy(ds,ctx.solid.data(),ctx.N*sizeof(bool),cudaMemcpyHostToDevice));
    dim3 b(16,16), g((ctx.nx+15)/16,(ctx.ny+15)/16);
    kmatvec<<<g,b>>>(dp,dAp,ds,ctx.nx,ctx.ny,ctx.stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    std::vector<double> gpu(ctx.N); cudaMemcpy(gpu.data(),dAp,ctx.N*sizeof(double),cudaMemcpyDeviceToHost);
    double d=max_diff(cpu.data(),gpu.data(),ctx.N,(bool*)ctx.solid.data());
    printf("  max|cpu-gpu|=%e  %s\n",d,d<1e-14?"PASS":"FAIL");
    cudaFree(dp);cudaFree(dAp);cudaFree(ds); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 4: Restriction
// ═══════════════════════════════════════════════════════════
static int test_restrict(TestCtx& ctx) {
    printf("Test 4: Restriction\n");
    int fnx=ctx.nx,fny=ctx.ny,fs=ctx.stride,cnx=fnx/2,cny=fny/2,cs=cnx+2,Nc=(cnx+2)*(cny+2);
    std::vector<double> xf(ctx.N),bf(ctx.N),bc_cpu(Nc,0);
    std::vector<char> sf(ctx.N,0),sc(Nc,0);
    for(int i=1;i<=fnx;i++)for(int j=1;j<=fny;j++){xf[i+j*fs]=std::sin(i*0.5);bf[i+j*fs]=1.0;}

    cpu_rbgs_full(xf.data(),bf.data(),(bool*)sf.data(),fnx,fny,fs,ctx.idx2,ctx.idy2,ctx.diag);
    // Now restrict: coarse.b = avg(b_fine - A*x_fine)
    for(int ic=1;ic<=cnx;ic++) for(int jc=1;jc<=cny;jc++){
        int if_=2*ic-1,jf_=2*jc-1;double sum=0;int cnt=0;
        for(int di=0;di<2;di++) for(int dj=0;dj<2;dj++){
            int fi=if_+di,fj=jf_+dj,fiid=fi+fj*fs;
            if(sf[fiid])continue;
            double pC=xf[fiid];
            double pL=(fi>1&&!sf[(fi-1)+fj*fs])?xf[(fi-1)+fj*fs]:pC;
            double pR=(fi<fnx&&!sf[(fi+1)+fj*fs])?xf[(fi+1)+fj*fs]:pC;
            double pB=(fj>1&&!sf[fi+(fj-1)*fs])?xf[fi+(fj-1)*fs]:pC;
            double pT=(fj<fny&&!sf[fi+(fj+1)*fs])?xf[fi+(fj+1)*fs]:pC;
            double lap=(pL+pR)*ctx.idx2+(pB+pT)*ctx.idy2;
            sum+=bf[fiid]-ctx.diag*pC+lap; cnt++;
        }
        int cid=ic+jc*cs; if(cnt>0)bc_cpu[cid]=sum/cnt;
    }

    double *dxf,*dbf,*dbc; bool *dsf,*dsc;
    CUDA_CHECK(cudaMalloc(&dxf,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(dxf,xf.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dbf,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(dbf,bf.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dbc,Nc*sizeof(double)));     CUDA_CHECK(cudaMemset(dbc,0,Nc*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dsf,ctx.N*sizeof(bool)));    CUDA_CHECK(cudaMemcpy(dsf,sf.data(),ctx.N*sizeof(bool),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dsc,Nc*sizeof(bool)));       CUDA_CHECK(cudaMemset(dsc,0,Nc*sizeof(bool)));
    dim3 b(16,16), g((cnx+15)/16,(cny+15)/16);
    krestrict<<<g,b>>>(dxf,dbf,dsf,dbc,dsc,fnx,fny,fs,cs,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    std::vector<double> gpu(Nc); cudaMemcpy(gpu.data(),dbc,Nc*sizeof(double),cudaMemcpyDeviceToHost);
    double d=max_diff(bc_cpu.data(),gpu.data(),Nc);
    printf("  max|cpu-gpu|=%e  %s\n",d,d<1e-14?"PASS":"FAIL");
    cudaFree(dxf);cudaFree(dbf);cudaFree(dbc);cudaFree(dsf);cudaFree(dsc); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 5: Prolongation
// ═══════════════════════════════════════════════════════════
static int test_prolong(TestCtx& ctx) {
    printf("Test 5: Prolongation\n");
    int cnx=ctx.nx/2,cny=ctx.ny/2,cs=cnx+2,fs=ctx.stride,Nc=(cnx+2)*(cny+2);
    std::vector<double> xf_cpu(ctx.N,0),xc(Nc,0);
    std::vector<char> sf(ctx.N,0);
    for(int i=1;i<=cnx;i++)for(int j=1;j<=cny;j++)xc[i+j*cs]=(i+j)*0.25;
    for(int i=1;i<=ctx.nx;i++)for(int j=1;j<=ctx.ny;j++)if(i==3&&j>=2&&j<=5)sf[i+j*fs]=1;
    // CPU
    for(int i=1;i<=ctx.nx;i++) for(int j=1;j<=ctx.ny;j++){
        int fid=i+j*fs; if(sf[fid])continue; int ic=(i+1)/2,jc=(j+1)/2; xf_cpu[fid]+=2.0*xc[ic+jc*cs];
    }
    // GPU
    double *dxf,*dxc; bool *dsf;
    CUDA_CHECK(cudaMalloc(&dxf,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemset(dxf,0,ctx.N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dxc,Nc*sizeof(double)));    CUDA_CHECK(cudaMemcpy(dxc,xc.data(),Nc*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dsf,ctx.N*sizeof(bool)));   CUDA_CHECK(cudaMemcpy(dsf,sf.data(),ctx.N*sizeof(bool),cudaMemcpyHostToDevice));
    dim3 b(16,16), g((ctx.nx+15)/16,(ctx.ny+15)/16);
    kprolong<<<g,b>>>(dxf,dxc,dsf,ctx.nx,ctx.ny,fs,cs);
    cudaDeviceSynchronize();
    std::vector<double> gpu(ctx.N); cudaMemcpy(gpu.data(),dxf,ctx.N*sizeof(double),cudaMemcpyDeviceToHost);
    double d=max_diff(xf_cpu.data(),gpu.data(),ctx.N,(bool*)sf.data());
    printf("  max|cpu-gpu|=%e  %s\n",d,d<1e-14?"PASS":"FAIL");
    cudaFree(dxf);cudaFree(dxc);cudaFree(dsf); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 6: Dot product
// ═══════════════════════════════════════════════════════════
static int test_dot(TestCtx& ctx) {
    printf("Test 6: Dot product\n");
    double cpu_s = cpu_dot(ctx.b.data(),ctx.b.data(),(bool*)ctx.solid.data(),ctx.N);
    double *da,*db,*dpart; bool *ds;
    CUDA_CHECK(cudaMalloc(&da,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(da,ctx.b.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&db,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(db,ctx.b.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ds,ctx.N*sizeof(bool)));   CUDA_CHECK(cudaMemcpy(ds,ctx.solid.data(),ctx.N*sizeof(bool),cudaMemcpyHostToDevice));
    int nb=(ctx.N+255)/256; CUDA_CHECK(cudaMalloc(&dpart,nb*sizeof(double)));
    kdot<<<nb,256>>>(da,db,ds,ctx.N,dpart); cudaDeviceSynchronize();
    double gpu_s=gpu_dot_reduce(dpart,nb);
    double d=std::abs(cpu_s-gpu_s);
    printf("  cpu=%.15e gpu=%.15e diff=%e  %s\n",cpu_s,gpu_s,d,d<1e-14?"PASS":"FAIL");
    cudaFree(da);cudaFree(db);cudaFree(ds);cudaFree(dpart); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 7: AXPY
// ═══════════════════════════════════════════════════════════
static int test_axpy(TestCtx& ctx) {
    printf("Test 7: AXPY\n");
    std::vector<double> cpu=ctx.b, x=ctx.b; for(auto& v:x)v*=0.5;
    cpu_axpy(cpu.data(),x.data(),0.3,(bool*)ctx.solid.data(),ctx.N);

    double *dy,*dx; bool *ds;
    CUDA_CHECK(cudaMalloc(&dy,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(dy,ctx.b.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dx,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(dx,x.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ds,ctx.N*sizeof(bool)));   CUDA_CHECK(cudaMemcpy(ds,ctx.solid.data(),ctx.N*sizeof(bool),cudaMemcpyHostToDevice));
    kaxpy<<<(ctx.N+255)/256,256>>>(dy,dx,0.3,ds,ctx.N); cudaDeviceSynchronize();
    std::vector<double> gpu(ctx.N); cudaMemcpy(gpu.data(),dy,ctx.N*sizeof(double),cudaMemcpyDeviceToHost);
    double d=max_diff(cpu.data(),gpu.data(),ctx.N,(bool*)ctx.solid.data());
    printf("  max|cpu-gpu|=%e  %s\n",d,d<1e-14?"PASS":"FAIL");
    cudaFree(dy);cudaFree(dx);cudaFree(ds); return d<1e-14?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 8: Mean & subtract_mean
// ═══════════════════════════════════════════════════════════
static int test_mean(TestCtx& ctx) {
    printf("Test 8: Mean & subtract_mean\n");
    double cpu_m=cpu_mean(ctx.b.data(),(bool*)ctx.solid.data(),ctx.N);
    std::vector<double> cpu_v=ctx.b; for(int k=0;k<ctx.N;k++) if(!ctx.solid[k])cpu_v[k]-=cpu_m;

    double *dv,*dpart; bool *ds; int *dcnt;
    CUDA_CHECK(cudaMalloc(&dv,ctx.N*sizeof(double))); CUDA_CHECK(cudaMemcpy(dv,ctx.b.data(),ctx.N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ds,ctx.N*sizeof(bool)));   CUDA_CHECK(cudaMemcpy(ds,ctx.solid.data(),ctx.N*sizeof(bool),cudaMemcpyHostToDevice));
    int nb=(ctx.N+255)/256; CUDA_CHECK(cudaMalloc(&dpart,nb*sizeof(double))); CUDA_CHECK(cudaMalloc(&dcnt,nb*sizeof(int)));

    ksum<<<nb,256>>>(dv,ds,ctx.N,dpart); cudaDeviceSynchronize();
    double gpu_sum=gpu_dot_reduce(dpart,nb);
    kcount<<<nb,256>>>(ds,ctx.N,dcnt); cudaDeviceSynchronize();
    int gpu_cnt=gpu_count_reduce(dcnt,nb);
    double gpu_m=gpu_cnt>0?gpu_sum/gpu_cnt:0;
    ksubmean<<<(ctx.N+255)/256,256>>>(dv,gpu_m,ds,ctx.N); cudaDeviceSynchronize();
    std::vector<double> gpu_v(ctx.N); cudaMemcpy(gpu_v.data(),dv,ctx.N*sizeof(double),cudaMemcpyDeviceToHost);

    double md=std::abs(cpu_m-gpu_m);
    double vd=max_diff(cpu_v.data(),gpu_v.data(),ctx.N,(bool*)ctx.solid.data());
    printf("  |cpu_mean-gpu_mean|=%e  max|v_cpu-v_gpu|=%e  %s\n",md,vd,(md<1e-14&&vd<1e-14)?"PASS":"FAIL");
    cudaFree(dv);cudaFree(ds);cudaFree(dpart);cudaFree(dcnt); return (md<1e-14&&vd<1e-14)?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Test 9: One full PCG iteration (all kernels chained)
// ═══════════════════════════════════════════════════════════
static int test_pcg_one_iter(TestCtx& ctx) {
    printf("Test 9: One PCG iteration (GPU vs CPU)\n");
    int N=ctx.N,nx=ctx.nx,ny=ctx.ny,stride=ctx.stride;
    int nb=(N+255)/256;
    dim3 b2d(16,16),g2d((nx+15)/16,(ny+15)/16);

    // Prepare RHS (negated, zero-mean — as PCG expects internally)
    double m=cpu_mean(ctx.b.data(),(bool*)ctx.solid.data(),N);
    std::vector<double> rhs(N); for(int k=0;k<N;k++) if(!ctx.solid[k]) rhs[k]=-(ctx.b[k]-m);

    // ── CPU: one PCG iteration ──
    std::vector<double> cp(N,0), cr=rhs, cz(N), cp_vec(N), cAp(N);
    // z0 = M^{-1}*r0
    cpu_rbgs_full(cz.data(),cr.data(),(bool*)ctx.solid.data(),nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    cp_vec=cz; double crsold=cpu_dot(cr.data(),cz.data(),(bool*)ctx.solid.data(),N);
    // iter 0
    cpu_matvec(cp_vec.data(),cAp.data(),(bool*)ctx.solid.data(),nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    double cpAp=cpu_dot(cp_vec.data(),cAp.data(),(bool*)ctx.solid.data(),N);
    double calpha=crsold/cpAp;
    cpu_axpy(cp.data(),cp_vec.data(),calpha,(bool*)ctx.solid.data(),N);
    cpu_axpy(cr.data(),cAp.data(),-calpha,(bool*)ctx.solid.data(),N);
    // z1
    std::fill(cz.begin(),cz.end(),0);
    cpu_rbgs_full(cz.data(),cr.data(),(bool*)ctx.solid.data(),nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    double crz=cpu_dot(cr.data(),cz.data(),(bool*)ctx.solid.data(),N);
    double cbeta=crz/crsold; crsold=crz;
    // p1 = z1 + beta * p0
    std::vector<double> cpu_p1=cz; cpu_axpy(cpu_p1.data(),cp_vec.data(),cbeta,(bool*)ctx.solid.data(),N);

    // ── GPU: one PCG iteration ──
    double *dg_r,*dg_z,*dg_p,*dg_Ap,*dg_part; bool *dg_s; int *dg_cnt;
    CUDA_CHECK(cudaMalloc(&dg_r,N*sizeof(double)));   CUDA_CHECK(cudaMemcpy(dg_r,rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dg_z,N*sizeof(double)));   CUDA_CHECK(cudaMemset(dg_z,0,N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dg_p,N*sizeof(double)));   CUDA_CHECK(cudaMemset(dg_p,0,N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dg_Ap,N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dg_s,N*sizeof(bool)));     CUDA_CHECK(cudaMemcpy(dg_s,ctx.solid.data(),N*sizeof(bool),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dg_part,nb*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dg_cnt,nb*sizeof(int)));

    // z0 = RBGS(r0)  (1 sweep)
    krbgs1<<<g2d,b2d>>>(dg_z,dg_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    krbgs2<<<g2d,b2d>>>(dg_z,dg_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(dg_p,dg_z,N*sizeof(double),cudaMemcpyDeviceToDevice)); // p0 = z0

    // rsold = dot(r, z)
    kdot<<<nb,256>>>(dg_r,dg_z,dg_s,N,dg_part); cudaDeviceSynchronize();
    double grsold=gpu_dot_reduce(dg_part,nb);

    // matvec(p, Ap)
    kmatvec<<<g2d,b2d>>>(dg_p,dg_Ap,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    kdot<<<nb,256>>>(dg_p,dg_Ap,dg_s,N,dg_part); cudaDeviceSynchronize();
    double gpAp=gpu_dot_reduce(dg_part,nb);

    double galpha=grsold/gpAp;
    // x += alpha*p, r -= alpha*Ap
    kaxpy<<<(N+255)/256,256>>>(dg_p,dg_p,galpha-1,dg_s,N); // hack: p was z0, now we want p = galpha*z0 ...
    // Wait, gpu p starts at 0. x += alpha * p means: x_new = 0 + alpha * p. So p should be alpha * z0.
    // Let me redo: cp = 0 + calpha * cp_vec. GPU: dg_p is p (not used as accumulator).
    // Actually dg_p is used as the accumulator. Let me restart properly.
    cudaMemset(dg_p,0,N*sizeof(double));
    kaxpy<<<(N+255)/256,256>>>(dg_p,dg_z,galpha,dg_s,N);      // cp = alpha * z0
    kaxpy<<<(N+255)/256,256>>>(dg_r,dg_Ap,-galpha,dg_s,N);     // cr -= alpha * Ap
    cudaDeviceSynchronize();

    // z1 = RBGS(r1)
    cudaMemset(dg_z,0,N*sizeof(double));
    krbgs1<<<g2d,b2d>>>(dg_z,dg_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();
    krbgs2<<<g2d,b2d>>>(dg_z,dg_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag);
    cudaDeviceSynchronize();

    kdot<<<nb,256>>>(dg_r,dg_z,dg_s,N,dg_part); cudaDeviceSynchronize();
    double grz=gpu_dot_reduce(dg_part,nb);
    double gbeta=grz/grsold;

    // p1 = z1 + beta * p0 (= dg_z already has z0, no wait — dg_z was z1. p0 is dg_z before the second RBGS. But we overwrote dg_z.)
    // We need to save p0. Let me re-read what I did above.
    // Actually: cudaMemcpy(dg_p, dg_z, ...) saved z0 as p0 (before any PCG operations on z).
    // Then I did matvec on dg_p (which is z0). Then I corrupted dg_p (memset+axpy set it to alpha*z0).
    // So p0 = dg_z at the start = z0. But dg_z was overwritten by the second RBGS.
    // For p1 = z1 + beta * p0, I need z1 (dg_z after second RBGS) and p0 (original z0, now lost).
    // I need an extra buffer. Let me use dg_Ap as temporary.

    // Actually, the flow should be:
    // 1. dg_r = rhs, dg_z = 0, dg_p = 0
    // 2. RBGS(dg_z, dg_r) → z0
    // 3. Copy dg_z → dg_p → p0 = z0
    // 4. matvec(dg_p, dg_Ap) → Ap0
    // 5. axpy(dg_p, dg_z, alpha) — wrong! dg_p IS z0, I want x1 = 0 + alpha*z0 = alpha*dg_p
    // 6. axpy(dg_r, dg_Ap, -alpha) → r1
    // 7. Zero dg_z, RBGS(dg_z, dg_r) → z1
    // 8. beta = dot(r1,z1)/dot(r0,z0)
    // 9. p1 = z1 + beta * p0. But p0 was corrupted in step 5!

    // Let me fix: save p0 separately.
    // Actually p0 = z0 = original dg_z. I should save it before corrupting.
    // Let me use dg_Ap as p0 storage after matvec.
    // No wait, let me just simplify the test.

    // Save p0 (dg_z = z0) into temporary
    cudaMemcpy(dg_Ap,dg_z,N*sizeof(double),cudaMemcpyDeviceToDevice); // dg_Ap = p0 = z0 (sacrificing dg_Ap)

    // Now restart properly:
    cudaMemcpy(dg_r,rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemset(dg_p,0,N*sizeof(double));
    cudaMemset(dg_z,0,N*sizeof(double));

    // z0 = RBGS(r0)
    krbgs1<<<g2d,b2d>>>(dg_z,dg_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();
    krbgs2<<<g2d,b2d>>>(dg_z,dg_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();
    kdot<<<nb,256>>>(dg_r,dg_z,dg_s,N,dg_part); cudaDeviceSynchronize();
    grsold=gpu_dot_reduce(dg_part,nb);

    // Save p0
    cudaMemcpy(dg_Ap,dg_z,N*sizeof(double),cudaMemcpyDeviceToDevice); // dg_Ap = p0

    // Ap0 = matvec(p0)
    kmatvec<<<g2d,b2d>>>(dg_Ap,dg_p,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); // dg_p = Ap0 TEMP
    cudaDeviceSynchronize();
    kdot<<<nb,256>>>(dg_Ap,dg_p,dg_s,N,dg_part); cudaDeviceSynchronize();
    gpAp=gpu_dot_reduce(dg_part,nb);
    galpha=grsold/gpAp;

    // Now dg_p holds Ap0, dg_Ap holds p0
    // x1 = alpha * p0 → zero dg_p and use it as accumulator
    cudaMemset(dg_p,0,N*sizeof(double)); // dg_p = x now
    kaxpy<<<(N+255)/256,256>>>(dg_p,dg_Ap,galpha,dg_s,N); // x = alpha * p0
    // r1 = r0 - alpha * Ap0
    kaxpy<<<(N+255)/256,256>>>(dg_r,dg_p,-1.0,dg_s,N); // wait, dg_p is x, not Ap0. I lost Ap0!

    // This is getting messy. Let me just use 4 separate device arrays.
    // For the test, I'll allocate them.

    cudaFree(dg_r);cudaFree(dg_z);cudaFree(dg_p);cudaFree(dg_Ap);

    double *d_r,*d_z,*d_p0,*d_ap;
    CUDA_CHECK(cudaMalloc(&d_r,N*sizeof(double)));   CUDA_CHECK(cudaMemcpy(d_r,rhs.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_z,N*sizeof(double)));   CUDA_CHECK(cudaMemset(d_z,0,N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p0,N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ap,N*sizeof(double)));

    // z0 = RBGS(r0)
    krbgs1<<<g2d,b2d>>>(d_z,d_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();
    krbgs2<<<g2d,b2d>>>(d_z,d_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();
    cudaMemcpy(d_p0,d_z,N*sizeof(double),cudaMemcpyDeviceToDevice); // p0 = z0

    kdot<<<nb,256>>>(d_r,d_z,dg_s,N,dg_part); cudaDeviceSynchronize();
    grsold=gpu_dot_reduce(dg_part,nb);

    // Ap0 = A*p0
    kmatvec<<<g2d,b2d>>>(d_p0,d_ap,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();
    kdot<<<nb,256>>>(d_p0,d_ap,dg_s,N,dg_part); cudaDeviceSynchronize();
    gpAp=gpu_dot_reduce(dg_part,nb);
    galpha=grsold/gpAp;

    // x1 = alpha * p0 (start from 0)
    cudaMemset(d_z,0,N*sizeof(double)); // reuse d_z as x
    kaxpy<<<(N+255)/256,256>>>(d_z,d_p0,galpha,dg_s,N); // x = alpha * p0 (in d_z)

    // r1 = r0 - alpha * Ap0
    kaxpy<<<(N+255)/256,256>>>(d_r,d_ap,-galpha,dg_s,N); cudaDeviceSynchronize();

    // z1 = RBGS(r1)
    cudaMemset(d_ap,0,N*sizeof(double)); // reuse d_ap as z1
    krbgs1<<<g2d,b2d>>>(d_ap,d_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();
    krbgs2<<<g2d,b2d>>>(d_ap,d_r,dg_s,nx,ny,stride,ctx.idx2,ctx.idy2,ctx.diag); cudaDeviceSynchronize();

    kdot<<<nb,256>>>(d_r,d_ap,dg_s,N,dg_part); cudaDeviceSynchronize();
    grz=gpu_dot_reduce(dg_part,nb);
    gbeta=grz/grsold;

    // Compare intermediate values
    printf("  CPU rsold=%.10e  GPU rsold=%.10e  diff=%e\n",crsold,grsold,std::abs(crsold-grsold));
    printf("  CPU pAp=%.10e    GPU pAp=%.10e    diff=%e\n",cpAp,gpAp,std::abs(cpAp-gpAp));
    printf("  CPU alpha=%.10e GPU alpha=%.10e diff=%e\n",calpha,galpha,std::abs(calpha-galpha));
    printf("  CPU beta=%.10e  GPU beta=%.10e  diff=%e\n",cbeta,gbeta,std::abs(cbeta-gbeta));

    // Compare p1: p1 = z1 + beta*p0 (stored in cpu_p1)
    // GPU: need to compute p1 = z1 + beta * p0
    kaxpy<<<(N+255)/256,256>>>(d_ap,d_p0,gbeta,dg_s,N); cudaDeviceSynchronize(); // d_ap = z1 + beta * p0
    std::vector<double> gpu_p1(N);
    cudaMemcpy(gpu_p1.data(),d_ap,N*sizeof(double),cudaMemcpyDeviceToHost);

    double d_p1=max_diff(cpu_p1.data(),gpu_p1.data(),N,(bool*)ctx.solid.data());
    printf("  max|cpu_p1-gpu_p1|=%e  %s\n",d_p1,d_p1<1e-12?"PASS":"FAIL");

    cudaFree(d_r);cudaFree(d_z);cudaFree(d_p0);cudaFree(d_ap);cudaFree(dg_s);cudaFree(dg_part);cudaFree(dg_cnt);
    return d_p1<1e-12?1:0;
}

// ═══════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════
int main() {
    TestCtx ctx; ctx.init();
    int p=0,f=0;
    (test_rbgs_pass1(ctx) ? p : f)++;
    (test_rbgs_pass2(ctx) ? p : f)++;
    (test_matvec(ctx)     ? p : f)++;
    (test_restrict(ctx)   ? p : f)++;
    (test_prolong(ctx)    ? p : f)++;
    (test_dot(ctx)        ? p : f)++;
    (test_axpy(ctx)       ? p : f)++;
    (test_mean(ctx)       ? p : f)++;
    (test_pcg_one_iter(ctx)? p : f)++;
    printf("\nCUDA full tests: %d pass, %d fail\n",p,f);
    return f>0;
}

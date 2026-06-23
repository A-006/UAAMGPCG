// GPU port of the semi-structured matrix-free multigrid-PCG solver, validated
// against the CPU reference. The algebraically-consistent hierarchy (Galerkin
// CSR levels + octree aggregation) is built on the host with the verified CPU
// code; the hot loop — PCG with a multigrid V-cycle preconditioner — runs
// entirely on the GPU: CSR SpMV, weighted-Jacobi smoothing (parallel
// replacement for the CPU's symmetric Gauss-Seidel), and scatter/gather
// restriction/prolongation through the aggregation map.
//
// Validation: GPU PCG solution == CPU PCG solution (both solve the same SPD
// system) to ~1e-6, on a uniform grid and an adaptive narrow-band grid.
#include "semistruct/adaptive_grid_2d.h"
#include "semistruct/poisson_operator_2d.h"
#include "semistruct/multigrid_2d.h"
#include "semistruct/pcg_2d.h"
#include "semistruct/sdf_2d.h"
#include "../test_utils.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace semistruct;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return 2; } } while (0)

// ── device kernels ──
__global__ void kSpmv(int n, const int* rp, const int* col, const double* val,
                      const double* x, double* y) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double s = 0;
    for (int p = rp[r]; p < rp[r + 1]; ++p) s += val[p] * x[col[p]];
    y[r] = s;
}
__global__ void kResidual(int n, const double* b, const double* Ax, double* r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - Ax[i];
}
__global__ void kJacobi(int n, const int* rp, const int* col, const double* val,
                        const double* diag, const double* x, const double* b,
                        double* xnew, double omega) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    if (diag[r] == 0.0) { xnew[r] = x[r]; return; }
    double Ax = 0;
    for (int p = rp[r]; p < rp[r + 1]; ++p) Ax += val[p] * x[col[p]];
    xnew[r] = x[r] + omega * (b[r] - Ax) / diag[r];
}
__global__ void kRestrict(int nf, const int* agg, const double* res, double* bc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nf) atomicAdd(&bc[agg[i]], res[i]);
}
__global__ void kProlong(int nf, const int* agg, const double* xc, double* x, double beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nf) x[i] += beta * xc[agg[i]];
}
__global__ void kAxpy(int n, double a, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
__global__ void kXpay(int n, double a, const double* x, double* y) {  // y = x + a*y
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + a * y[i];
}
__global__ void kDot(int n, const double* a, const double* b, double* acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(acc, a[i] * b[i]);
}
__global__ void kCopy(int n, const double* a, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

static int grid1(int n) { return (n + 255) / 256; }

// device level
struct DLevel {
    int n, nnz;
    int *rp, *col; double *val, *diag;
    int *agg;       // size n (maps to next coarser), null on coarsest
    double *x, *b, *r, *tmp;
};

struct GpuMG {
    std::vector<DLevel> L;
    double omega = 0.6; int nu1 = 2, nu2 = 2, ncoarse = 60; double beta = 2.0;
    double* d_acc;  // scalar accumulator

    void upload(const Multigrid2D& mg) {
        int nlev = mg.A.size();
        L.resize(nlev);
        for (int k = 0; k < nlev; ++k) {
            const CSR& A = mg.A[k];
            DLevel& d = L[k];
            d.n = A.n; d.nnz = A.val.size();
            cudaMalloc(&d.rp, (A.n + 1) * sizeof(int));
            cudaMalloc(&d.col, d.nnz * sizeof(int));
            cudaMalloc(&d.val, d.nnz * sizeof(double));
            cudaMalloc(&d.diag, A.n * sizeof(double));
            cudaMemcpy(d.rp, A.rowptr.data(), (A.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d.col, A.col.data(), d.nnz * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d.val, A.val.data(), d.nnz * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d.diag, A.diag.data(), A.n * sizeof(double), cudaMemcpyHostToDevice);
            for (double** p : {&d.x, &d.b, &d.r, &d.tmp}) cudaMalloc(p, A.n * sizeof(double));
            d.agg = nullptr;
            if (k + 1 < nlev) {
                cudaMalloc(&d.agg, A.n * sizeof(int));
                cudaMemcpy(d.agg, mg.agg[k].data(), A.n * sizeof(int), cudaMemcpyHostToDevice);
            }
        }
        cudaMalloc(&d_acc, sizeof(double));
    }
    void free() {
        for (auto& d : L) { cudaFree(d.rp); cudaFree(d.col); cudaFree(d.val); cudaFree(d.diag);
            cudaFree(d.x); cudaFree(d.b); cudaFree(d.r); cudaFree(d.tmp); if (d.agg) cudaFree(d.agg); }
        cudaFree(d_acc);
    }
    void smooth(int k, int sweeps) {
        DLevel& d = L[k];
        for (int s = 0; s < sweeps; ++s) {
            kJacobi<<<grid1(d.n), 256>>>(d.n, d.rp, d.col, d.val, d.diag, d.x, d.b, d.tmp, omega);
            std::swap(d.x, d.tmp);
        }
    }
    void vcycle(int k) {
        DLevel& d = L[k];
        if (k == (int)L.size() - 1) { smooth(k, ncoarse); return; }
        smooth(k, nu1);
        kSpmv<<<grid1(d.n), 256>>>(d.n, d.rp, d.col, d.val, d.x, d.tmp);
        kResidual<<<grid1(d.n), 256>>>(d.n, d.b, d.tmp, d.r);
        DLevel& c = L[k + 1];
        cudaMemset(c.b, 0, c.n * sizeof(double));
        cudaMemset(c.x, 0, c.n * sizeof(double));
        kRestrict<<<grid1(d.n), 256>>>(d.n, d.agg, d.r, c.b);
        vcycle(k + 1);
        kProlong<<<grid1(d.n), 256>>>(d.n, d.agg, c.x, d.x, beta);
        smooth(k, nu2);
    }
    void apply(const double* r, double* z) {  // device pointers, level-0 sized
        kCopy<<<grid1(L[0].n), 256>>>(L[0].n, r, L[0].b);
        cudaMemset(L[0].x, 0, L[0].n * sizeof(double));
        vcycle(0);
        kCopy<<<grid1(L[0].n), 256>>>(L[0].n, L[0].x, z);
    }
    double dot(const double* a, const double* b, int n) {
        double zero = 0; cudaMemcpy(d_acc, &zero, sizeof(double), cudaMemcpyHostToDevice);
        kDot<<<grid1(n), 256>>>(n, a, b, d_acc);
        double h; cudaMemcpy(&h, d_acc, sizeof(double), cudaMemcpyDeviceToHost);
        return h;
    }
};

// GPU PCG on the finest level; returns iters, writes solution to host x.
static int gpuPCG(GpuMG& mg, const CSR& A0, const std::vector<double>& b,
                  std::vector<double>& x, int max_iter, double rel_tol) {
    int n = A0.n;
    double *dx, *dr, *dz, *dp, *dAp;
    cudaMalloc(&dx, n * sizeof(double)); cudaMalloc(&dr, n * sizeof(double));
    cudaMalloc(&dz, n * sizeof(double)); cudaMalloc(&dp, n * sizeof(double));
    cudaMalloc(&dAp, n * sizeof(double));
    cudaMemset(dx, 0, n * sizeof(double));
    cudaMemcpy(dr, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    DLevel& F = mg.L[0];
    double res0 = std::sqrt(mg.dot(dr, dr, n));
    mg.apply(dr, dz);
    kCopy<<<grid1(n), 256>>>(n, dz, dp);
    double rz = mg.dot(dr, dz, n);
    int it = 0;
    for (; it < max_iter; ++it) {
        kSpmv<<<grid1(n), 256>>>(n, F.rp, F.col, F.val, dp, dAp);
        double pAp = mg.dot(dp, dAp, n);
        double alpha = rz / pAp;
        kAxpy<<<grid1(n), 256>>>(n, alpha, dp, dx);
        kAxpy<<<grid1(n), 256>>>(n, -alpha, dAp, dr);
        double rn = std::sqrt(mg.dot(dr, dr, n));
        if (rn <= rel_tol * res0) { ++it; break; }
        mg.apply(dr, dz);
        double rz_new = mg.dot(dr, dz, n);
        double beta = rz_new / rz;
        kXpay<<<grid1(n), 256>>>(n, beta, dz, dp);  // dp = dz + beta*dp
        rz = rz_new;
    }
    x.resize(n);
    cudaMemcpy(x.data(), dx, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dr); cudaFree(dz); cudaFree(dp); cudaFree(dAp);
    return it;
}

static void runCase(AdaptiveGrid2D& g, const char* tag, int& fails) {
    g.bc = {BC::Dirichlet, BC::Dirichlet, BC::Dirichlet, BC::Dirichlet};
    PoissonOperator2D op; op.build(g);
    Multigrid2D mg; mg.build(op, Coarsening::Algebraic);
    auto forcing = [](double x, double y) { return std::sin(4 * x) * std::cos(3 * y) + 1.0; };
    std::vector<double> b = op.rhs(forcing, [](double, double) { return 0.0; });
    // CPU reference
    std::vector<double> xc;
    auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
    PCGResult Rc = pcg(op.A, b, xc, pre, 500, 1e-10, false);
    // GPU
    GpuMG gmg; gmg.upload(mg);
    std::vector<double> xg;
    int git = gpuPCG(gmg, op.A, b, xg, 500, 1e-10);
    gmg.free();
    double diff = 0, ref = 0;
    for (int i = 0; i < op.A.n; ++i) { diff = std::max(diff, std::fabs(xg[i] - xc[i])); ref = std::max(ref, std::fabs(xc[i])); }
    printf("  %-14s DOFs=%-6d CPU iters=%d GPU iters=%d  max|xg-xc|=%.3e (|x|~%.2e)\n",
           tag, op.A.n, Rc.iters, git, diff, ref);
    if (!(diff < 1e-6 * (ref + 1e-30))) { printf("  [FAIL] %s GPU==CPU\n", tag); fails++; }
    else printf("  [PASS] %s GPU solution == CPU solution\n", tag);
}

int main() {
    test_header("Semi-structured multigrid-PCG on GPU (2D)");
    int dev = 0;
    if (cudaSetDevice(dev) != cudaSuccess) { printf("  [SKIP] no CUDA device\n"); return 0; }
    int fails = 0;
    { AdaptiveGrid2D g; g.build(1, 64, refineUniform());                          runCase(g, "uniform-64", fails); }
    { AdaptiveGrid2D g; g.build(3, 16, refineNarrowBand(0.5, 0.5, 0.25, 0.08));   runCase(g, "narrow-band", fails); }
    printf("\n  %s\n", fails == 0 ? "ALL GPU CASES PASS" : "SOME GPU CASES FAILED");
    return fails == 0 ? 0 : 1;
}

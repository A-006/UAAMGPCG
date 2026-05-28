#pragma once
#include "solver/cuda/cuda_common_3d.h"
#include "solver/cuda/cuda_uaamg_preconditioner_3d.h"
#include <memory>

/// CUDA 3D PCG solver with UAAMG preconditioner.
class CudaPCG3D {
public:
    ~CudaPCG3D() { free_buffers(); }
    void solve(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol);

    /// Optimized solve: fused dot products + shared-memory tiled matvec.
    void solve_optimized(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol);

    /// Mixed-precision solve: FP64 CG outer loop + FP32 Galerkin preconditioner.
    /// The preconditioner is only an approximation, so FP32 is fine; the FP64 CG
    /// vectors/dots keep orthogonality and convergence to 1e-6. ~2× via the FP32
    /// V-cycle while still converging (pure FP32 PCG diverges).
    void solve_mixed(CudaGrid3D& g, double* p, double* rhs, int max_iter, double tol);

    /// Fully device-resident PCG: all scalars/alpha/beta on device, no per-iter
    /// host sync (cudaMemcpy). Fixed iteration count (like the paper). Removes the
    /// per-dot host round-trip that bottlenecks small grids.
    void solve_device(CudaGrid3D& g, double* p, double* rhs, int iters, double tol);

private:
    std::unique_ptr<CudaUAAMGPreconditioner3D> precond_{
        std::make_unique<CudaUAAMGPreconditioner3D>()};
    // FP32 preconditioner + float grid + buffers (mixed-precision path).
    std::unique_ptr<CudaUAAMGPreconditioner3Dt> precond_f_{
        std::make_unique<CudaUAAMGPreconditioner3Dt>()};
    CudaGrid3Df gf_{}; int gf_N_ = 0; float *d_rf = nullptr, *d_zf = nullptr;
    bool mixed_ = false;
    void mixed_apply(int N, const double* dr, double* dz);

    double *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    double *d_dot_buf = nullptr, *d_scalar = nullptr;
    int    *d_count_buf = nullptr;
    size_t dot_buf_size_ = 0;
    int N_ = 0;

    void ensure_buffers(int N);
    void free_buffers();

public:
    CudaUAAMGPreconditioner3D& precond() { return *precond_; }
    int last_iters = 0;     // PCG iters actually performed in the last solve_optimized call
    double last_rel_res = 1.0;  // sqrt(rsnew / r0_sq) at the moment of break
};

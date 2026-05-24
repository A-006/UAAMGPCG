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

private:
    std::unique_ptr<CudaUAAMGPreconditioner3D> precond_{
        std::make_unique<CudaUAAMGPreconditioner3D>()};
    double *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    double *d_dot_buf = nullptr;
    int    *d_count_buf = nullptr;
    size_t dot_buf_size_ = 0;
    int N_ = 0;

    void ensure_buffers(int N);
    void free_buffers();

public:
    CudaUAAMGPreconditioner3D& precond() { return *precond_; }
};

#pragma once
#include "solver/cuda/cuda_common.h"

/// CUDA CG solver (no preconditioner, M=I).
/// Minimal implementation — just PCG with identity preconditioner.
class CudaCG {
public:
    ~CudaCG() { free_buffers(); }

    /// Solve (-nabla^2) p = rhs on device.
    void solve(CudaGrid& g, double* p, double* rhs, int max_iter, double tol);

private:
    double *d_r = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    double *d_dot_buf = nullptr;
    int    *d_count_buf = nullptr;
    size_t dot_buf_size_ = 0;
    int N_ = 0;

    void ensure_buffers(int N);
    void free_buffers();
};

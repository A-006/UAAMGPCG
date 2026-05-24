#pragma once
#include "solver/cuda/cuda_common.h"
#include "solver/cuda/cuda_uaamg_preconditioner.h"
#include <memory>

class CudaPCG {
public:
    ~CudaPCG() { free_buffers(); }
    void solve(CudaGrid& g, double* p, double* rhs, int max_iter, double tol);

private:
    std::unique_ptr<CudaUAAMGPreconditioner> precond_{
        std::make_unique<CudaUAAMGPreconditioner>()};
    double *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    double *d_dot_buf = nullptr;
    int    *d_count_buf = nullptr;
    size_t dot_buf_size_ = 0;
    int N_ = 0;

    void ensure_buffers(int N);
    void free_buffers();

public:
    CudaUAAMGPreconditioner& precond() { return *precond_; }
};

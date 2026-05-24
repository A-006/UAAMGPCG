#include "solver/cuda_pcg_solver.h"

CudaPCGSolver::CudaPCGSolver(bool use_precond)
    : use_precond_(use_precond)
{
    if (use_precond_)
        pcg_ = std::make_unique<CudaPCG>();
    else
        cg_ = std::make_unique<CudaCG>();
}

CudaPCGSolver::~CudaPCGSolver() {
    if (d_p_)   cudaFree(d_p_);
    if (d_rhs_) cudaFree(d_rhs_);
}

std::string CudaPCGSolver::name() const {
    return use_precond_ ? "PCG(CUDA-UAAMG)" : "CG(CUDA)";
}

void CudaPCGSolver::solve(Grid& g, const std::vector<double>& rhs,
                           int max_iter, double tol) {
    int N = (int)rhs.size();
    int nx = g.nx, ny = g.ny;

    // Allocate / reallocate GPU buffers if size or dimensions changed
    bool realloc = (!gpu_grid_ || N_ != N);
    bool resize  = (gpu_grid_ && (gpu_grid_->nx != nx || gpu_grid_->ny != ny));
    if (realloc) {
        if (d_p_)   cudaFree(d_p_);
        if (d_rhs_) cudaFree(d_rhs_);
        gpu_grid_ = std::make_unique<CudaGrid>();
        gpu_grid_->allocate(nx, ny, g.dx, g.dy);
        cudaMalloc(&d_p_,   N * sizeof(double));
        cudaMalloc(&d_rhs_, N * sizeof(double));
        N_ = N;
        resize = false; // already fresh
    }
    if (resize) {
        gpu_grid_->free();
        gpu_grid_->allocate(nx, ny, g.dx, g.dy);
    }

    // Copy solid mask: CPU vector<bool> is bit-packed → expand to char array
    if (realloc || resize) {
        std::vector<char> solid_tmp(N);
        for (int k = 0; k < N; k++) solid_tmp[k] = g.solid[k];
        cudaMemcpy(gpu_grid_->solid, solid_tmp.data(), N, cudaMemcpyHostToDevice);
    }

    // Copy RHS → GPU
    cudaMemcpy(d_rhs_, rhs.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Solve on GPU
    cudaMemset(d_p_, 0, N * sizeof(double));
    if (use_precond_)
        pcg_->solve(*gpu_grid_, d_p_, d_rhs_, max_iter, tol);
    else
        cg_->solve(*gpu_grid_, d_p_, d_rhs_, max_iter, tol);

    // Copy pressure back to CPU
    cudaMemcpy(g.p.data(), d_p_, N * sizeof(double), cudaMemcpyDeviceToHost);
}

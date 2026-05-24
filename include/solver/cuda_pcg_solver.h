#pragma once
#include "solver/solver.h"
#include "solver/cuda/cuda_common.h"
#include "solver/cuda/cuda_pcg.h"
#include "solver/cuda/cuda_cg.h"
#include <memory>

/// GPU PCG solver — wraps GPU solver to implement the CPU Solver interface.
/// Copies RHS/pressure between CPU Grid and GPU CudaGrid each solve step.
///
/// Two modes:
///   - use_precond=true  → CudaPCG (UAAMG preconditioned)
///   - use_precond=false → CudaCG  (identity, plain CG)
class CudaPCGSolver : public Solver {
public:
    explicit CudaPCGSolver(bool use_precond = true);
    ~CudaPCGSolver() override;

    void solve(Grid& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override;

    CudaUAAMGPreconditioner& precond() { return pcg_->precond(); }

private:
    bool use_precond_;
    std::unique_ptr<CudaGrid> gpu_grid_;
    double *d_p_ = nullptr, *d_rhs_ = nullptr;
    int N_ = 0;
    std::unique_ptr<CudaPCG> pcg_;
    std::unique_ptr<CudaCG>  cg_;
};

#pragma once
#include "solver/solver_3d.h"
#include "solver/cuda/cuda_common_3d.h"
#include "solver/cuda/cuda_pcg_3d.h"
#include "solver/cuda/cuda_cg_3d.h"
#include <memory>

/// GPU 3D PCG solver — wraps GPU solver to implement the Solver3D interface.
/// Copies RHS/pressure between CPU Grid3D and GPU CudaGrid3D each solve step.
///
/// Two modes:
///   - use_precond=true  → CudaPCG3D (UAAMG preconditioned)
///   - use_precond=false → CudaCG3D  (identity, plain CG)
class CudaPCGSolver3D : public Solver3D {
public:
    explicit CudaPCGSolver3D(bool use_precond = true);
    ~CudaPCGSolver3D() override;

    void solve(Grid3D& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override;

    CudaUAAMGPreconditioner3D& precond() { return pcg_->precond(); }

private:
    bool use_precond_;
    std::unique_ptr<CudaGrid3D> gpu_grid_;
    double *d_p_ = nullptr, *d_rhs_ = nullptr;
    int N_ = 0;
    std::unique_ptr<CudaPCG3D> pcg_;
    std::unique_ptr<CudaCG3D>  cg_;
};

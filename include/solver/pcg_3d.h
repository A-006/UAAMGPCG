/**
 * @file pcg_3d.h
 * @brief Preconditioned Conjugate Gradient solver for 3D.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/solver_3d.h"
#include "solver/preconditioner/preconditioner_3d.h"
#include <memory>

/**
 * @brief Preconditioned Conjugate Gradient solver for 3D.
 *
 * CG accelerated by a pluggable preconditioner \f$M^{-1}\f$.
 * With IdentityPreconditioner3D this reduces to standard CG.
 * The preconditioner is owned and its lifetime is tied to the solver.
 */
class PCG3D : public Solver3D {
public:
    /**
     * @brief Construct PCG3D with a preconditioner.
     * @param p Preconditioner instance (ownership transferred).
     */
    explicit PCG3D(std::unique_ptr<Preconditioner3D> p);

    /**
     * @brief Solve using PCG.
     * @param g        3D MAC grid.
     * @param rhs      Right-hand side (zero-mean is enforced internally).
     * @param max_iter Maximum PCG iterations.
     * @param tol      Convergence tolerance on residual L2-norm.
     */
    void solve(Grid3D& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;

    /** @brief Returns "PCG3D(<preconditioner>)". */
    std::string name() const override;

private:
    std::unique_ptr<Preconditioner3D> precond_;  ///< Preconditioner instance.
};

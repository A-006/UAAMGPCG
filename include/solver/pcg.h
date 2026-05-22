/**
 * @file pcg.h
 * @brief Preconditioned Conjugate Gradient solver.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "solver/solver.h"
#include "solver/preconditioner/preconditioner.h"
#include <memory>

/**
 * @brief Preconditioned Conjugate Gradient solver.
 *
 * CG accelerated by a pluggable preconditioner \f$M^{-1}\f$.
 * With IdentityPreconditioner this reduces to standard CG.
 * The preconditioner is owned and its lifetime is tied to the solver.
 */
class PCG : public Solver {
public:
    /**
     * @brief Construct PCG with a preconditioner.
     * @param p Preconditioner instance (ownership transferred).
     */
    explicit PCG(std::unique_ptr<Preconditioner> p);

    /**
     * @brief Solve using PCG.
     * @param g        MAC grid.
     * @param rhs      Right-hand side (zero-mean is enforced internally).
     * @param max_iter Maximum PCG iterations.
     * @param tol      Convergence tolerance on residual L2-norm.
     */
    void solve(Grid& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;

    /** @brief Returns "PCG(<preconditioner>)". */
    std::string name() const override;

private:
    std::unique_ptr<Preconditioner> precond_;  ///< Preconditioner instance.
};

/**
 * @file amgx_solver.h
 * @brief NVIDIA AMGX as a drop-in MAC-grid Poisson backend.
 * @author liutao
 * @date 2026-06-23
 *
 * Wraps the AMGX C API behind the project's Solver interface. The pressure
 * Poisson system (A = -nabla^2 with the same Neumann/solid handling as PCG)
 * is assembled on the host, solved on the GPU by AMGX, and the result is
 * scattered back into g.p. This mirrors how AMGX is used as a drop-in linear
 * solver inside CFD packages (e.g. ANSYS Fluent).
 *
 * AMGX details are hidden behind a pimpl so this header has no AMGX dependency
 * and the rest of the project compiles unchanged. When the project is built
 * without AMGX support (UAAMG_WITH_AMGX=OFF), constructing this solver throws
 * a clear runtime_error.
 */
#pragma once
#include "solver/solver_2d.h"
#include <memory>

/**
 * @brief Poisson solver backed by NVIDIA AMGX (algebraic multigrid + Krylov).
 *
 * The AMGX solver configuration defaults to classical-AMG-preconditioned PCG.
 * Override it by pointing the environment variable UAAMG_AMGX_CONFIG at an
 * AMGX JSON config file. Set UAAMG_AMGX_VERBOSE=1 to print AMGX solve stats.
 */
class AMGXSolver : public Solver {
public:
    AMGXSolver();
    ~AMGXSolver() override;

    /**
     * @brief Solve nabla^2 p = rhs on the GPU via AMGX. Modifies g.p in place.
     * @param g        MAC grid (pressure modified on exit; zero-mean enforced).
     * @param rhs      Right-hand side (same convention as the other solvers).
     * @param max_iter Maximum AMGX iterations.
     * @param tol      Relative residual tolerance (RELATIVE_INI, L2).
     */
    void solve(Grid& g, const std::vector<double>& rhs, int max_iter, double tol) override;

    /** @brief Returns "AMGX". */
    std::string name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

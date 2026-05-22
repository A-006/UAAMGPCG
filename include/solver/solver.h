/**
 * @file solver.h
 * @brief Abstract solver interface for Poisson equation on MAC grid.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "core/grid.h"
#include <vector>
#include <string>

/**
 * @brief Abstract interface for Poisson equation \f$\nabla^2 p = rhs\f$ on MAC grid.
 *
 * All solvers are matrix-free — no sparse matrix is assembled.
 * Each derived class implements a specific iterative method.
 */
class Solver {
public:
    virtual ~Solver() = default;

    /**
     * @brief Solve \f$\nabla^2 p = rhs\f$. Modifies g.p in-place.
     * @param g        MAC grid (pressure modified on exit).
     * @param rhs      Right-hand side vector.
     * @param max_iter Maximum iterations (or sweeps for Jacobi/RBGS).
     * @param tol      Convergence tolerance (for CG/PCG).
     */
    virtual void solve(Grid& g, const std::vector<double>& rhs,
                       int max_iter, double tol) = 0;

    /** @brief Human-readable solver name. */
    virtual std::string name() const = 0;
};

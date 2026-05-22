#pragma once
#include "core/grid.h"
#include <vector>
#include <string>

// Abstract interface for Poisson equation ∇²p = rhs on MAC grid.
// All solvers are matrix-free (no sparse matrix assembled).
class PoissonSolver {
public:
    virtual ~PoissonSolver() = default;

    // Solve ∇²p = rhs. Modifies g.p in-place.
    // max_iter: maximum iterations (or sweeps for Jacobi/RBGS)
    // tol:      convergence tolerance (for CG/PCG)
    virtual void solve(Grid& g, const std::vector<double>& rhs,
                       int max_iter, double tol) = 0;

    virtual std::string name() const = 0;
};

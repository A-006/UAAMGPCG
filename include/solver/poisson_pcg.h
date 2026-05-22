#pragma once
#include "solver/poisson_solver.h"

class PCGSolver : public PoissonSolver {
public:
    void solve(Grid& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override { return "PCG"; }
};

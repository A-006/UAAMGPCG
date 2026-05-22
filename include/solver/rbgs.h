/**
 * @file rbgs.h
 * @brief Red-Black Gauss-Seidel iterative solver for the Poisson equation.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "solver/solver.h"

/**
 * @brief Red-Black Gauss-Seidel solver.
 *
 * Alternates between red (i+j even) and black (i+j odd) points.
 * Each half-sweep uses the latest neighbor values, combining the
 * convergence rate of GS with the parallelism of Jacobi.
 * The tol parameter is ignored — always runs max_iter sweeps.
 */
class RBGS : public Solver {
public:
    /**
     * @brief Run RBGS iterations.
     * @param g        MAC grid.
     * @param rhs      Right-hand side.
     * @param max_iter Number of RBGS sweeps.
     * @param tol      Ignored.
     */
    void solve(Grid& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override { return "RBGS"; }
};

/**
 * @file jacobi.h
 * @brief Jacobi iterative solver for the Poisson equation.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "solver/solver.h"

/**
 * @brief Jacobi (damped) iterative solver.
 *
 * Uses the inverse diagonal as a fixed-point iterator.
 * Handles Neumann null-space by subtracting the mean after each sweep.
 * The tol parameter is ignored — always runs max_iter sweeps.
 */
class Jacobi : public Solver {
public:
    /**
     * @brief Run Jacobi iterations.
     * @param g        MAC grid.
     * @param rhs      Right-hand side.
     * @param max_iter Number of Jacobi sweeps.
     * @param tol      Ignored.
     */
    void solve(Grid& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override { return "Jacobi"; }
};

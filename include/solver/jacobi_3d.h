/**
 * @file jacobi_3d.h
 * @brief Jacobi iterative solver for the 3D Poisson equation.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/solver_3d.h"

/**
 * @brief Jacobi (damped) iterative solver for 3D.
 *
 * Uses the inverse diagonal as a fixed-point iterator.
 * Handles Neumann null-space by subtracting the mean after each sweep.
 * The tol parameter is ignored — always runs max_iter sweeps.
 */
class Jacobi3D : public Solver3D {
public:
    /**
     * @brief Run Jacobi iterations.
     * @param g        3D MAC grid.
     * @param rhs      Right-hand side.
     * @param max_iter Number of Jacobi sweeps.
     * @param tol      Ignored.
     */
    void solve(Grid3D& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override { return "Jacobi3D"; }
};

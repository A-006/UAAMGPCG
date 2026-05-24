/**
 * @file rbgs_3d.h
 * @brief Red-Black Gauss-Seidel iterative solver for the 3D Poisson equation.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/solver_3d.h"

/**
 * @brief Red-Black Gauss-Seidel solver for 3D.
 *
 * Alternates between red ((i+j+k) even) and black ((i+j+k) odd) points.
 * Each half-sweep uses the latest neighbor values, combining the
 * convergence rate of GS with the parallelism of Jacobi.
 * The tol parameter is ignored — always runs max_iter sweeps.
 */
class RBGS3D : public Solver3D {
public:
    /**
     * @brief Run RBGS iterations.
     * @param g        3D MAC grid.
     * @param rhs      Right-hand side.
     * @param max_iter Number of RBGS sweeps.
     * @param tol      Ignored.
     */
    void solve(Grid3D& g, const std::vector<double>& rhs,
               int max_iter, double tol) override;
    std::string name() const override { return "RBGS3D"; }
};

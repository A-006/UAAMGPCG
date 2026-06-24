/**
 * @file amgx_solver_3d.h
 * @brief NVIDIA AMGX as a drop-in 3D MAC-grid Poisson backend.
 * @author liutao
 * @date 2026-06-23
 *
 * 3D twin of AMGXSolver. Assembles the 7-point pressure Poisson system from a
 * Grid3D (A = -nabla^2 with the same Neumann/solid handling as PCG3D), solves it
 * on the GPU via AMGX, and scatters the result into g.p. Used on the CPU 3D
 * pipeline (LFMSimulator3D) via Factory3D when solver=amgx; the simulation loop
 * runs on the host while AMGX does the linear solve on the GPU.
 *
 * AMGX details live behind the shared AmgxBackend, so this header has no AMGX
 * dependency. Built without AMGX support, constructing it throws.
 */
#pragma once
#include "solver/solver_3d.h"
#include <memory>

/**
 * @brief 3D Poisson solver backed by NVIDIA AMGX.
 *
 * Defaults to classical-AMG-preconditioned PCG. Override via UAAMG_AMGX_CONFIG
 * (AMGX JSON config file); set UAAMG_AMGX_VERBOSE=1 to print AMGX solve stats.
 */
class AMGXSolver3D : public Solver3D {
public:
    AMGXSolver3D();
    ~AMGXSolver3D() override;

    /**
     * @brief Solve nabla^2 p = rhs on the GPU via AMGX. Modifies g.p in place.
     * @param g        3D MAC grid (pressure modified on exit; zero-mean enforced).
     * @param rhs      Right-hand side (same convention as the other solvers).
     * @param max_iter Maximum AMGX iterations.
     * @param tol      Relative residual tolerance (RELATIVE_INI, L2).
     */
    void solve(Grid3D& g, const std::vector<double>& rhs, int max_iter, double tol) override;

    /** @brief Returns "AMGX3D". */
    std::string name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

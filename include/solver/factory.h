/**
 * @file factory.h
 * @brief Solver factory — creates solvers by name.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "solver/solver.h"
#include <memory>
#include <string>

/**
 * @brief Factory that creates Solver instances from a string key.
 *
 * Supported keys:
 *   "jacobi"  — Jacobi
 *   "rbgs"    — Red-Black Gauss-Seidel
 *   "cg"      — PCG with IdentityPreconditioner (plain CG)
 *   "pcg"     — PCG with GMGPreconditioner
 *   "pcg_gmg" — PCG with GMGPreconditioner
 *   "pcg_amg" — PCG with AMGPreconditioner
 */
class Factory {
public:
    /**
     * @brief Create a solver by name.
     * @param name Solver key (see class doc).
     * @return Unique pointer to the solver. Defaults to Jacobi on unknown key.
     */
    static std::unique_ptr<Solver> create(const std::string& name);
};

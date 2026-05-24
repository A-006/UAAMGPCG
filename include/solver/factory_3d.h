/**
 * @file factory_3d.h
 * @brief 3D solver factory — creates Solver3D instances by name.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/solver_3d.h"
#include <memory>
#include <string>

/**
 * @brief Factory that creates Solver3D instances from a string key.
 *
 * Supported keys:
 *   "jacobi"    — Jacobi3D
 *   "rbgs"      — RBGS3D
 *   "cg"        — PCG3D with IdentityPreconditioner3D (plain CG)
 *   "pcg"       — PCG3D with GMGPreconditioner3D
 *   "pcg_gmg"   — PCG3D with GMGPreconditioner3D
 *   "pcg_amg"   — PCG3D with AMGPreconditioner3D
 *   "pcg_uaamg" — PCG3D with UAAMGPreconditioner3D
 */
class Factory3D {
public:
    /**
     * @brief Create a 3D solver by name.
     * @param name Solver key (see class doc).
     * @return Unique pointer to the solver. Defaults to Jacobi3D on unknown key.
     */
    static std::unique_ptr<Solver3D> create(const std::string& name);
};

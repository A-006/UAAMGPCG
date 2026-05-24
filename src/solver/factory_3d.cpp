/**
 * @file factory_3d.cpp
 * @brief 3D solver factory — maps string keys to solver + preconditioner combinations.
 * @author liutao
 * @date 2026-05-24
 */
#include "solver/factory_3d.h"
#include "solver/jacobi_3d.h"
#include "solver/rbgs_3d.h"
#include "solver/pcg_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
// 3D preconditioners — included once implemented
// #include "solver/preconditioner/gmg_preconditioner_3d.h"
// #include "solver/preconditioner/amg_preconditioner_3d.h"
// #include "solver/preconditioner/uaamg_preconditioner_3d.h"

std::unique_ptr<Solver3D> Factory3D::create(const std::string& name) {
    if (name == "jacobi")    return std::make_unique<Jacobi3D>();
    if (name == "rbgs")      return std::make_unique<RBGS3D>();
    if (name == "cg")        return std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>());
    // if (name == "pcg")       return std::make_unique<PCG3D>(std::make_unique<GMGPreconditioner3D>());
    // if (name == "pcg_gmg")   return std::make_unique<PCG3D>(std::make_unique<GMGPreconditioner3D>());
    // if (name == "pcg_amg")   return std::make_unique<PCG3D>(std::make_unique<AMGPreconditioner3D>());
    // if (name == "pcg_uaamg") return std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>());
    // Fallback: return Jacobi3D for pcg/pcg_gmg/pcg_amg/pcg_uaamg until 3D preconditioners exist
    if (name == "pcg" || name == "pcg_gmg" || name == "pcg_amg" || name == "pcg_uaamg")
        return std::make_unique<Jacobi3D>();
    return std::make_unique<Jacobi3D>();
}

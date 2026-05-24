/**
 * @file factory.cpp
 * @brief Solver factory — maps string keys to solver + preconditioner combinations.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/factory.h"
#include "solver/jacobi.h"
#include "solver/rbgs.h"
#include "solver/pcg.h"
#include "solver/preconditioner/identity_preconditioner.h"
#include "solver/preconditioner/gmg_preconditioner.h"
#include "solver/preconditioner/amg_preconditioner.h"
#include "solver/preconditioner/uaamg_preconditioner.h"

std::unique_ptr<Solver> Factory::create(const std::string& name) {
    if (name == "jacobi")   return std::make_unique<Jacobi>();
    if (name == "rbgs")     return std::make_unique<RBGS>();
    if (name == "cg")       return std::make_unique<PCG>(std::make_unique<IdentityPreconditioner>());
    if (name == "pcg")      return std::make_unique<PCG>(std::make_unique<GMGPreconditioner>());
    if (name == "pcg_gmg")  return std::make_unique<PCG>(std::make_unique<GMGPreconditioner>());
    if (name == "pcg_amg")  return std::make_unique<PCG>(std::make_unique<AMGPreconditioner>());
    if (name == "pcg_uaamg") return std::make_unique<PCG>(std::make_unique<UAAMGPreconditioner>());
    return std::make_unique<Jacobi>();
}

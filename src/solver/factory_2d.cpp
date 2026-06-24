/**
 * @file factory.cpp
 * @brief Solver factory — maps string keys to solver + preconditioner combinations.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/factory_2d.h"
#include "solver/amgx/amgx_solver_2d.h"
#include "solver/relaxation/jacobi_2d.h"
#include "solver/relaxation/rbgs_2d.h"
#include "solver/krylov/pcg_2d.h"
#include "solver/preconditioner/identity_preconditioner_2d.h"
#include "solver/preconditioner/gmg_preconditioner_2d.h"
#include "solver/preconditioner/amg_preconditioner_2d.h"
#include "solver/preconditioner/uaamg_preconditioner_2d.h"
#include "util/registry.h"

namespace {

// Maps a solver key to a (solver + preconditioner) combination. Adding a
// solver is one line here; see also the 3D twin in factory_3d.cpp.
util::Registry<Solver>& registry() {
    static auto reg = [] {
        util::Registry<Solver> r;
        r.add("jacobi", [] { return std::make_unique<Jacobi>(); });
        r.add("rbgs", [] { return std::make_unique<RBGS>(); });
        r.add("cg", [] { return std::make_unique<PCG>(std::make_unique<IdentityPreconditioner>()); });
        r.add("pcg", [] { return std::make_unique<PCG>(std::make_unique<GMGPreconditioner>()); });
        r.add("pcg_gmg", [] { return std::make_unique<PCG>(std::make_unique<GMGPreconditioner>()); });
        r.add("pcg_amg", [] { return std::make_unique<PCG>(std::make_unique<AMGPreconditioner>()); });
        r.add("pcg_uaamg", [] { return std::make_unique<PCG>(std::make_unique<UAAMGPreconditioner>()); });
        r.add("amgx", [] { return std::make_unique<AMGXSolver>(); });
        return r;
    }();
    return reg;
}

} // namespace

std::unique_ptr<Solver> Factory::create(const std::string& name) {
    auto& reg = registry();
    return reg.contains(name) ? reg.create(name) : std::make_unique<Jacobi>(); // unknown ⇒ Jacobi
}

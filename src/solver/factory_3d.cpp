/**
 * @file factory_3d.cpp
 * @brief 3D solver factory — maps string keys to solver + preconditioner combinations.
 * @author liutao
 * @date 2026-05-24
 */
#include "solver/factory_3d.h"
#include "solver/amgx/amgx_solver_3d.h"
#include "solver/relaxation/jacobi_3d.h"
#include "solver/relaxation/rbgs_3d.h"
#include "solver/krylov/pcg_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include "solver/preconditioner/gmg_preconditioner_3d.h"
#include "solver/preconditioner/amg_preconditioner_3d.h"
#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include "util/registry.h"

namespace {

util::Registry<Solver3D>& registry() {
    static auto reg = [] {
        util::Registry<Solver3D> r;
        r.add("jacobi", [] { return std::make_unique<Jacobi3D>(); });
        r.add("rbgs", [] { return std::make_unique<RBGS3D>(); });
        r.add("cg",
              [] { return std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>()); });
        // Preconditioned CG variants. "pcg" defaults to the paper's UAAMG.
        r.add("pcg", [] { return std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>()); });
        r.add("pcg_gmg",
              [] { return std::make_unique<PCG3D>(std::make_unique<GMGPreconditioner3D>()); });
        r.add("pcg_amg",
              [] { return std::make_unique<PCG3D>(std::make_unique<AMGPreconditioner3D>()); });
        r.add("pcg_uaamg",
              [] { return std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>()); });
        r.add("amgx", [] { return std::make_unique<AMGXSolver3D>(); });
        return r;
    }();
    return reg;
}

} // namespace

std::unique_ptr<Solver3D> Factory3D::create(const std::string& name) {
    auto& reg = registry();
    // Unknown keys fall back to Jacobi3D.
    return reg.contains(name) ? reg.create(name) : std::make_unique<Jacobi3D>();
}

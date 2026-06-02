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
#include "util/registry.h"
// 3D multigrid preconditioners (gmg/amg/uaamg) are not wired in yet; the pcg*
// keys fall back to Jacobi3D below until they exist.

namespace {

util::Registry<Solver3D>& registry() {
    static auto reg = [] {
        util::Registry<Solver3D> r;
        r.add("jacobi", [] { return std::make_unique<Jacobi3D>(); });
        r.add("rbgs", [] { return std::make_unique<RBGS3D>(); });
        r.add("cg",
              [] { return std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>()); });
        return r;
    }();
    return reg;
}

} // namespace

std::unique_ptr<Solver3D> Factory3D::create(const std::string& name) {
    auto& reg = registry();
    // Unknown or not-yet-implemented (pcg*) keys fall back to Jacobi3D.
    return reg.contains(name) ? reg.create(name) : std::make_unique<Jacobi3D>();
}

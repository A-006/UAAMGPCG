#include "simulator/scenarios/2d/primitives_2d.h"
#include "numerics/bc/patches.h"
#include "simulator/scenarios/2d/karman.h"
#include "simulator/scenarios/2d/leapfrog.h"

// Built-in 2D primitives. Each builder reads its parameters from the Param2D
// namespace it was handed and delegates to the existing field/geometry math;
// the defaults reproduce the historical hardcoded scenarios bit-for-bit when a
// case file leaves a key unset.
namespace scenarios {

// ── Geometry ────────────────────────────────────────────────────────────────
const std::unordered_map<std::string, GeomBuilder>& geom_registry() {
    static const std::unordered_map<std::string, GeomBuilder> r = {
        // Immersed circular cylinder (Kármán obstacle). Falls back to the core
        // cyl_* Config fields so `cyl_cx=… cyl_R=…` keep working unprefixed.
        {"cylinder",
         [](Grid& g, const Config& cfg, const Param2D& p) {
             Karman k{p.d("cx", cfg.cyl_cx), p.d("cy", cfg.cyl_cy), p.d("R", cfg.cyl_R),
                      p.d("U", cfg.U_inf)};
             if (k.cyl_R > 0)
                 setup_karman_cylinder(g, k);
         }},
    };
    return r;
}

// ── Initial conditions ───────────────────────────────────────────────────────
const std::unordered_map<std::string, IcBuilder>& ic_registry() {
    static const std::unordered_map<std::string, IcBuilder> r = {
        // Uniform stream u = U everywhere (warm start for the wake).
        {"uniform_inflow",
         [](Grid& g, const Config& cfg, const Param2D& p) {
             set_uniform_inflow(g, p.d("U", cfg.U_inf));
         }},
        // Sinusoidal v-band behind the cylinder, breaks y-symmetry.
        {"wake_perturb",
         [](Grid& g, const Config& cfg, const Param2D& p) {
             Karman k{p.d("cx", cfg.cyl_cx), p.d("cy", cfg.cyl_cy), p.d("R", cfg.cyl_R),
                      p.d("U", cfg.U_inf)};
             seed_wake_perturbation(g, k, p.d("amplitude", 0.01));
         }},
        // Two counter-rotating Lamb-Oseen dipoles (leapfrog IC).
        {"vortex_dipole",
         [](Grid& g, const Config&, const Param2D& p) {
             VortexDipoles d;
             d.core   = p.d("core", d.core);
             d.gamma  = p.d("gamma", d.gamma);
             d.half_d = p.d("half_d", d.half_d);
             d.yc     = p.d("yc", d.yc);
             d.x0     = p.d("x0", d.x0);
             d.x1     = p.d("x1", d.x1);
             seed_vortex_dipoles(g, d);
         }},
    };
    return r;
}

// ── Body forces ──────────────────────────────────────────────────────────────
const std::unordered_map<std::string, ForceBuilder>& force_registry() {
    static const std::unordered_map<std::string, ForceBuilder> r = {
        // Constant upward buoyancy on every fluid cell (smoke).
        {"buoyancy",
         [](Grid& g, const Config& cfg, const Param2D& p) {
             double a = p.d("accel", 5.0);
             for (int i = 1; i <= g.nx; i++)
                 for (int j = 1; j <= g.ny; j++)
                     if (!g.is_solid(i, j))
                         g.v_at(i, j) += cfg.dt * a;
         }},
    };
    return r;
}

// ── Boundary stacks ──────────────────────────────────────────────────────────
const std::unordered_map<std::string, BcBuilder>& bc_registry() {
    static const std::unordered_map<std::string, BcBuilder> r = {
        {"karman", [](const Config& cfg) { return bc::karman(cfg.U_inf); }},
        {"smoke", [](const Config&) { return bc::smoke(); }},
        {"free_slip_walls", [](const Config&) { return bc::free_slip_walls(); }},
    };
    return r;
}

} // namespace scenarios

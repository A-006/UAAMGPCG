#include "simulator/scenarios/karman_scenario.h"
#include "numerics/bc/patches.h"
#include "simulator/scenarios/karman.h"
#include <algorithm>

namespace scenarios {

void KarmanScenario::configure(Config& cfg) const {
    cfg.Lx      = 4.0;
    cfg.Ly      = 1.0;
    cfg.NY      = std::max(cfg.NX / 4, 16);
    cfg.dt      = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf; // CFL ≈ 0.5
    cfg.out_dir = "output_karman";
}

void KarmanScenario::init_grid(Grid& g, const Config& cfg) const {
    Karman k{cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R, cfg.U_inf};
    if (k.cyl_R > 0)
        setup_karman_cylinder(g, k);
    set_uniform_inflow(g, k.U_inf);
    // Break the y-symmetry — without this seed a perfectly symmetric setup
    // produces a standing symmetric vortex pair, never the alternating street.
    seed_wake_perturbation(g, k);
}

bc::BoundaryManager KarmanScenario::boundary_manager(const Config& cfg) const {
    return bc::karman(cfg.U_inf);
}

} // namespace scenarios

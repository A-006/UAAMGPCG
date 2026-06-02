#include "simulator/scenarios/2d/smoke_scenario.h"
#include "numerics/bc/patches.h"

namespace scenarios {

void SmokeScenario::configure(Config& cfg) const {
    cfg.Lx      = 1.0;
    cfg.Ly      = 1.0;
    cfg.NY      = cfg.NX;
    cfg.dt      = 0.005;
    cfg.out_dir = "output_smoke";
}

void SmokeScenario::init_grid(Grid&, const Config&) const {
    // Starts from rest in an empty box; nothing to seed.
}

bc::BoundaryManager SmokeScenario::boundary_manager(const Config&) const {
    return bc::smoke();
}

void SmokeScenario::apply_body_force(Grid& g, const Config& cfg) const {
    // Constant upward buoyancy on every fluid cell.
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i, j))
                g.v_at(i, j) += cfg.dt * 5.0;
}

} // namespace scenarios

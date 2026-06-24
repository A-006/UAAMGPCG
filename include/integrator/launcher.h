#pragma once
#include "core/config.h"
#include "integrator/make_simulator_3d.h"
#include "integrator/scene_2d.h"
#include "io/3d/case_3d.h"
#include "integrator/simulation.h"
#include <memory>

// ── The cfdsim launcher: turn argv into a ready-to-run Simulation ────────────
// The 2D and 3D engines are separate type hierarchies, so picking between them
// is genuinely dimension-specific. These two functions are the ONE place that
// branch lives — everything downstream (main) talks to the common Simulation
// base, so it runs any scenario with a single polymorphic call.
namespace launcher {

// Build the right Config from argv: dim==3 goes through scene3d's presets,
// everything else through scene2d's. The dimensionality is data — peeked from
// the case file / CLI `dim` (default 2) — so the launcher knows nothing about
// scenario names. Throws on bad input.
inline Config build_config(int argc, char** argv) {
    if (scene3d::peek_dim(argc, argv) == "3")
        return scene3d::build_config(argc, argv);
    return scene2d::build_config(argc, argv);
}

// Build the matching simulator (3D GPU/CPU, or the 2D CPU pipeline) and return
// it through the common Simulation base.
inline std::unique_ptr<Simulation> make_simulation(Config& cfg) {
    if (cfg.dim == 3)
        return scene3d::make_simulator(cfg);
    return scene2d::make_simulator(cfg);
}

} // namespace launcher

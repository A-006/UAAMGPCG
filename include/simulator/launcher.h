#pragma once
#include "config/config.h"
#include "simulator/make_simulator_3d.h"
#include "simulator/scene_2d.h"
#include "simulator/scene_3d.h"
#include "simulator/simulation.h"
#include <memory>

// ── The cfdsim launcher: turn argv into a ready-to-run Simulation ────────────
// The 2D and 3D engines are separate type hierarchies, so picking between them
// is genuinely dimension-specific. These two functions are the ONE place that
// branch lives — everything downstream (main) talks to the common Simulation
// base, so it runs any scenario with a single polymorphic call.
namespace launcher {

// Build the right Config from argv: 3D scenarios go through scene3d's presets,
// everything else through scene2d's. Throws on bad input.
inline Config build_config(int argc, char** argv) {
    if (scene3d::is_3d_scenario(scene3d::peek_scenario(argc, argv)))
        return scene3d::build_config(argc, argv);
    return scene2d::build_config(argc, argv);
}

// Build the matching simulator (3D GPU/CPU, or the 2D CPU pipeline) and return
// it through the common Simulation base.
inline std::unique_ptr<Simulation> make_simulation(Config& cfg) {
    if (scene3d::is_3d_scenario(cfg.scenario))
        return scene3d::make_simulator(cfg);
    return scene2d::make_simulator(cfg);
}

} // namespace launcher

/**
 * @file src/cfdsim.cpp
 * @brief Entry point for the unified LFM launcher (GPU or CPU, every scene).
 *
 * One program for the whole project. The scenario and every parameter come from
 * an INI file and/or `key=value` arguments, so there is a single binary instead
 * of a runner per scene:
 *
 *   ./cfdsim inputs/vortex_collision.in            # scene from a file
 *   ./cfdsim inputs/vortex_collision.in NX=192     # file, with CLI overrides
 *   ./cfdsim scenario=vortex_ring NX=96 cycles=80  # no file, all on the CLI
 *   ./cfdsim scenario=vortex_ring backend=cpu      # force the CPU backend
 *   ./cfdsim scenario=karman NX=128 solver=pcg     # 2D (CPU) scene
 *
 * This file is ONLY the entry point — read it top to bottom and that is the
 * whole program. Each step delegates to a library module:
 *   scene3d::build_config   — scenario presets + your INI/CLI overrides
 *   scene3d::make_simulator — pick GPU/CPU, build it, set the initial condition
 *   sim3d::run              — run the cycles and write the output frames
 * (3D scenes — vortex_ring, vortex_collision, collision_paper, delta_wing,
 * vortex_reconnection, trefoil_knot — see scene_3d.h. Everything else is 2D.)
 */
#include "config/cli.h"
#include "config/config.h"
#include "simulator/factory.h"           // 2D path: SimulatorFactory::create
#include "simulator/make_simulator_3d.h" // scene3d::make_simulator
#include "simulator/runner.h"            // 2D path: sim::run
#include "simulator/runner_3d.h"         // sim3d::run
#include "simulator/scene_3d.h"          // scene3d::is_3d_scenario / build_config
#include <iostream>

// 2D scenarios (karman, smoke, …) use the shared CPU pipeline: the 2D registry
// configures the run and sim::run drives it.
static int run_2d(int argc, char** argv) {
    auto cfg = config::parse_cli(argc, argv);
    if (!cfg)
        return 1;
    auto sim = SimulatorFactory::create(*cfg);
    sim::run(*sim, *cfg);
    return 0;
}

int main(int argc, char** argv) {
    if (!scene3d::is_3d_scenario(scene3d::peek_scenario(argc, argv)))
        return run_2d(argc, argv);

    try {
        Config cfg = scene3d::build_config(argc, argv); // presets + your overrides
        auto sim   = scene3d::make_simulator(cfg);      // GPU if available, else CPU
        sim3d::run(*sim, cfg);                          // run cycles, write frames
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"
                  << "Usage: cfdsim [input.in] [key=value]...  (key=value overrides the file)\n"
                     "  scenarios: vortex_ring | vortex_collision | collision_paper | leapfrog_rings | delta_wing | "
                     "vortex_reconnection | trefoil_knot\n";
        return 1;
    }
    return 0;
}

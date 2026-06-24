/**
 * @file src/main.cpp
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
 * whole program. 2D and 3D scenes run through the SAME three steps because both
 * engines share the Simulation base; the launcher hides the dimension choice:
 *   launcher::build_config  — scenario presets + your INI/CLI overrides
 *   launcher::make_simulation — build the 2D or 3D simulator for this scenario
 *   Simulation::run         — run to completion and write the output frames
 * (3D scenes — vortex_ring, vortex_collision, collision_paper, delta_wing,
 * vortex_reconnection, trefoil_knot — see scene_3d.h. Everything else is 2D.)
 */
#include "core/config.h"
#include "simulator/launcher.h" // launcher::build_config / make_simulation
#include <iostream>

int main(int argc, char** argv) {
    try {
        Config cfg = launcher::build_config(argc, argv);
        auto sim   = launcher::make_simulation(cfg);
        sim->run(cfg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"
                  << "Usage: cfdsim [input.in] [key=value]...  (key=value overrides the file)\n"
                     "  3D: vortex_ring | vortex_collision | collision_paper | leapfrog_rings | "
                     "delta_wing | vortex_reconnection | trefoil_knot\n"
                     "  2D: scenario=NAME (e.g. karman)\n";
        return 1;
    }
    return 0;
}

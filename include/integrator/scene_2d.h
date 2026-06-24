#pragma once
#include "io/cli.h"
#include "core/config.h"
#include "integrator/factory.h"
#include "integrator/simulator_base.h"
#include <memory>

// ── 2D scene provisioning — the peer of scene3d for the launcher ─────────────
// The 2D engine is the shared CPU pipeline (ScenarioRegistry + SimulatorFactory),
// so these are thin. They exist so the launcher dispatches 2D and 3D through one
// uniform sceneNd interface (build_config + make_simulator) instead of two
// differently-named factories.
namespace scene2d {

// Assemble a Config from argv (config file + key=value overrides, CLI wins).
// Throws on an unreadable file / unknown scenario / key / bad value.
inline Config build_config(int argc, char** argv) {
    return config::build_config(argc, argv);
}

// Build the 2D simulator for cfg (picks integrator/solver via the registry).
inline std::unique_ptr<Simulator> make_simulator(const Config& cfg) {
    return SimulatorFactory::create(cfg);
}

} // namespace scene2d

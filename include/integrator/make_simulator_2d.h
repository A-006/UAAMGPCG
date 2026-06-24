#pragma once
#include "io/config.h"
#include "integrator/simulator_2d.h"
#include <memory>

// 2D simulator provisioning — the peer of make_simulator_3d. Picks the backend
// and builds the 2D simulator. 2D's "GPU" is a GPU Poisson solver injected into
// the CPU sim loop (not a GPU-resident simulator like 3D), so it is opt-in.
namespace scene2d {

std::unique_ptr<Simulator> make_simulator(Config& cfg);

} // namespace scene2d

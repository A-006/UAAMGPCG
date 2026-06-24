#pragma once
#include "core/config.h"
#include "integrator/simulator_3d.h"
#include <memory>

namespace scene3d {

// Build the simulator for cfg's scenario, with its initial condition and wall
// BCs already applied — ready for sim3d::run. Resolves cfg["backend"]:
//   auto → GPU if this build has CUDA and a device is present, else CPU
//   gpu  → GPU (falls back to CPU with a warning if none is available)
//   cpu  → the host LFMSimulator3D (slower; correctness reference)
// The resolved choice ("gpu"/"cpu") is written back into cfg["backend"].
// Throws std::runtime_error on an invalid backend or a failed cudaSetDevice.
std::unique_ptr<Simulator3D> make_simulator(Config& cfg);

} // namespace scene3d

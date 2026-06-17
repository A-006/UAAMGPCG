#pragma once
#include "config/config.h"
#include "core/grid_3d.h"
#include "numerics/bc/patches_3d.h"
#include <string>

// ── 3D scene definitions for the cfdsim launcher ────────────────────────────
// Everything the launcher needs to turn a scenario name + key=value overrides
// into a ready-to-run Config and an initialized Grid3D, backend-agnostic so the
// GPU and CPU simulators share the exact same presets and initial condition.
//
// Supported scenarios: vortex_ring, vortex_collision, collision_paper,
// delta_wing, vortex_reconnection, trefoil_knot. Anything else is a 2D scenario
// handled by the shared 2D CPU pipeline.
namespace scene3d {

// True if `name` is one of the 3D scenarios this launcher provisions.
bool is_3d_scenario(const std::string& name);

// Peek the `scenario=` assignment from argv (INI file + CLI) without throwing;
// returns the default "vortex_ring" if none is given or argv can't be parsed.
// Used to pick the 2D-vs-3D code path before full Config validation.
std::string peek_scenario(int argc, char** argv);

// Assemble a Config from argv: apply the scenario's per-scene presets, then the
// INI/CLI overrides on top (so any user-set field wins), then post-process
// derived fields (e.g. delta-wing freestream from angle of attack).
// Throws std::runtime_error on an unknown scenario / key / unreadable file.
Config build_config(int argc, char** argv);

// Write the scenario's initial condition onto the host grid. If cfg sets
// `ic_dir`, loads that exact staggered field instead (author cross-check).
void setup(Grid3D& g, const Config& cfg);

// Wall BCs for the CPU backend: freestream box for the delta wing, free-slip
// box (+ immersed solid) otherwise. The GPU simulator applies these internally.
bc::BoundaryManager3D make_cpu_bcs(const Config& cfg);

} // namespace scene3d

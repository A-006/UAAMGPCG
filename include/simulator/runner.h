#pragma once
#include "core/config.h"
#include "simulator/simulator_base.h"
#include <functional>
#include <vector>

// ──────────────────────────────────────────────────────────────────
// sim::run — the single driver loop shared by both time integrators.
//
// It is integrator-agnostic: it advances `sim.step()` until
// `sim.time()` reaches `cfg.t_end`, dumps a VTK frame every
// `cfg.frame_skip` steps, and invokes any user observers on those same
// frames. Tools plug diagnostics (e.g. force / Strouhal CSV) in through
// the observer list instead of forking the loop.
// ──────────────────────────────────────────────────────────────────
namespace sim {

// Invoked on every output frame: (frame index, simulator, config).
using Observer = std::function<void(int frame, const Simulator& sim, const Config& cfg)>;

struct RunOptions {
    bool write_vtk    = true;        // dump VTK every frame_skip steps
    bool print_banner = true;        // header + timing report
    std::vector<Observer> observers; // extra per-frame diagnostics
};

void run(Simulator& sim, const Config& cfg, const RunOptions& opts = {});

} // namespace sim

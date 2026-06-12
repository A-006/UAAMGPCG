#pragma once
#include "config/config.h"
#include "simulator/simulator_3d.h"
#include <string>

// ── sim3d::run — the cfdsim cycle loop, shared by the GPU and CPU backends ──
// Prints the run banner, creates cfg.out_dir, writes an initial frame, then
// advances `cfg["cycles"]` LFM reinitialization cycles, dumping a VTK frame
// (full VtkWriter3D or slim |omega|, per cfg["vtk_mode"]) every cfg.frame_skip
// cycles and an optional cell-centered velocity .raw when cfg["dump_vel"] is
// set. Operates only on the Simulator3D base, so the caller owns backend choice.
namespace sim3d {

// The banner reports cfg["backend"] ("gpu"/"cpu", as resolved by make_simulator)
// and, for the GPU, cfg["gpu"] (device index).
void run(Simulator3D& sim, const Config& cfg);

} // namespace sim3d

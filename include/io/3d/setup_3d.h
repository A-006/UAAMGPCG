#pragma once
#include "core/config.h"
#include "mesh/grid_3d.h"
#include "mesh/bc/patches_3d.h"

// ── 3D initial condition + wall BC provisioning ─────────────────────────────
// Write the case's initial condition onto the grid (data-driven: each `ic = …`
// source picks a primitive and reads its params), and build the matching wall
// BCs. The solver core knows no scenario names — only the case data.
namespace scene3d {

// Apply the case's declared IC sources to the grid (or load a raw field if
// cfg["ic_dir"] is set). Dumps the field if cfg["dump_ic_dir"] is set.
void setup(Grid3D& g, const Config& cfg);

// Wall BCs for the CPU backend: a freestream box when cfg.lfm_bc=="freestream",
// else a free-slip box (+ immersed solid).
bc::BoundaryManager3D make_cpu_bcs(const Config& cfg);

} // namespace scene3d

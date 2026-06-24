#pragma once
#include "mesh/grid_3d.h"
#include "mesh/bc/patches_3d.h"

// ── Uniform freestream: IC + wall boundary conditions ───────────────────────
// Generic helpers for an immersed solid in a uniform freestream (used e.g. by
// the delta-wing case, but scenario-agnostic). Set the freestream as an initial
// field, build the matching wall BCs, and load an external SDF geometry.
namespace scenarios {

// Set inflow velocity (uniform U_inf in +x) on the entire domain.
void set_uniform_inflow(Grid3D& g, double U_inf);

// Inflow/outflow channel BCs: inflow on x-min, zero-gradient on x-max,
// free-slip y/z walls, no-slip on any immersed solid.
bc::BoundaryManager3D inflow_outflow_bcs(double U_inf);

// Set a uniform freestream (Ux,Uy,Uz) over the whole field (initial condition).
void set_uniform_freestream(Grid3D& g, double Ux, double Uy, double Uz);

// Load an external geometry from a .npy signed-distance field (float32, C-order,
// shape = grid nx×ny×nz): mark every cell with sdf<0 as solid.
void load_sdf_solid(Grid3D& g, const std::string& npy_path);

// Prescribe the SAME freestream velocity (Ux,Uy,Uz) on all six walls (so inflow
// flux = outflow flux exactly → mass-balanced, well-posed pure-Neumann
// pressure) + no-slip on any immersed solid.
bc::BoundaryManager3D freestream_box_bcs(double Ux, double Uy, double Uz);

} // namespace scenarios

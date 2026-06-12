#pragma once
#include "core/grid_3d.h"
#include <string>

// ── Disk-friendly 3D output used by the cfdsim launcher ─────────────────────
// The full VtkWriter3D (velocity vectors + extra scalars) is ~0.5–4.5 GB/frame
// at high resolution; the iso-surface / volume renderers only need |omega|.
// These mirror the formats the former per-scene runners wrote.
namespace io3d {

// Slim VTK: vorticity-magnitude scalar only (structured points).
void write_vort_vtk(const Grid3D& g, int frame, const std::string& dir);

// Cell-centered velocity (float32, C-order i-slowest, nx*ny*nz) as
// vx/vy/vz_<frame>.raw, so |omega| can be computed with the SAME np.gradient
// operator as the author's vx_*.npy — the apples-to-apples cross-check.
void write_vel_raw(const Grid3D& g, int frame, const std::string& dir);

} // namespace io3d

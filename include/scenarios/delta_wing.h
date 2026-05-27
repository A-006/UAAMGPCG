#pragma once
#include "core/grid_3d.h"
#include "bc/patches_3d.h"

// ──────────────────────────────────────────────────────────────────
// Delta wing immersed obstacle + inflow BC (paper Fig. 1 left).
//
// The wing is a flat triangular plate at angle-of-attack α, sitting in
// the x–z plane (chord along x, span along z) and tilted up about the
// y-axis. We mark every cell whose center lies inside the (thickened)
// triangle as solid; the surrounding flow + free-slip ceiling/floor
// generates wingtip vortices off the swept leading edges.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct DeltaWing {
    double leading_x = 0.6;            // x of the apex
    double chord     = 0.8;            // root chord length (apex → trailing edge)
    double semi_span = 0.4;            // half of the wing span (z direction)
    double thickness = 0.02;           // wing half-thickness (y direction)
    double aoa_deg   = 12.0;           // angle of attack (rotation about z)
    double y_mid     = 0.5;            // wing mid-plane y position
};

// Mark grid cells inside the delta wing as solid.
void setup_delta_wing(Grid3D& g, const DeltaWing& wing);

// Set inflow velocity (uniform U_inf in +x) on the entire domain.
void set_uniform_inflow(Grid3D& g, double U_inf);

// Build a BoundaryManager3D for the delta-wing scenario:
//   inflow on x-min, zero-gradient on x-max, free-slip y/z walls,
//   no-slip on the immersed wing.
bc::BoundaryManager3D delta_wing_bcs(double U_inf);

}  // namespace scenarios

#pragma once
#include "mesh/grid_2d.h"

// ──────────────────────────────────────────────────────────────────
// Leapfrogging vortex-dipole initial condition (paper Fig. 10).
//
// Two co-axial counter-rotating Lamb-Oseen vortex pairs propagating along +x.
// Reused by both the data-driven IC primitive ("vortex_dipole") and the
// scenario-setup unit tests, so the math has exactly one home.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct VortexDipoles {
    double core   = 0.12; // Lamb-Oseen core radius a
    double gamma  = 1.0;  // |circulation| of each vortex
    double half_d = 0.30; // half-separation of each dipole in y
    double yc     = -1.0; // propagation axis; <0 ⇒ use Ly/2
    double x0     = 1.4;  // leading dipole x
    double x1     = 0.9;  // trailing dipole x
};

// Add the 4-vortex (two-dipole) velocity field onto the grid's MAC faces.
void seed_vortex_dipoles(Grid& g, const VortexDipoles& d);

} // namespace scenarios

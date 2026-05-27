#pragma once
#include "core/grid_3d.h"
#include <array>

// ──────────────────────────────────────────────────────────────────
// Analytical initial conditions for vortex-ring scenarios.
//
// add_vortex_ring(): superimposes a Gaussian-cored vortex ring's velocity
// field onto Grid3D's MAC faces by direct Biot-Savart integration over a
// discretized circular filament.
//
// Used for the "vortex ring head-on collision" test (paper Fig. 3) by
// stacking two rings with opposite circulation along a shared axis.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct VortexRing {
    std::array<double, 3> center;  // ring center (cx, cy, cz)
    std::array<double, 3> axis;    // unit normal of the ring plane
    double radius;                 // ring radius
    double core;                   // Gaussian core radius (regularization)
    double circulation;            // Γ (sign sets rotation direction)
    int    n_segments = 200;       // filament discretization
};

void add_vortex_ring(Grid3D& g, const VortexRing& vr);

}  // namespace scenarios

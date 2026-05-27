#pragma once
#include "core/grid_3d.h"
#include "bc/patches_3d.h"

// ──────────────────────────────────────────────────────────────────
// Wind turbine in uniform inflow (paper Fig. 2).
//
// Three slender rectangular blades rotate about a horizontal hub. At
// each step the rotor angle is advanced and the solid mask + the
// rotational velocity on the blade surface are re-applied. The
// downstream helical trail emerges from the moving wing-tip vortices.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct WindTurbine {
    std::array<double, 3> hub_center{ {1.0, 0.75, 1.0} };  // hub position
    double blade_radius = 0.45;          // tip radius
    double hub_radius   = 0.05;
    double chord        = 0.08;          // blade chord (along axis)
    double thickness    = 0.025;         // blade half-thickness in tangential dir
    int    n_blades     = 3;
    double angular_velocity = 4.0;       // rad/s — rotation about +x axis
};

// Set the solid mask AND prescribe the rotational velocity on solid
// faces according to the current rotor angle (rad).
void apply_turbine_state(Grid3D& g, const WindTurbine& wt, double angle);

// Build BCs for the turbine scenario (inflow / outflow / free-slip / solid).
bc::BoundaryManager3D wind_turbine_bcs(double U_inf);

}  // namespace scenarios

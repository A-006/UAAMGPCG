#pragma once
#include "core/grid.h"
#include "bc/boundary_condition.h"

// ──────────────────────────────────────────────────────────────────
// 2D Kármán vortex street (paper Fig. 8).
//
// Single immersed circular cylinder in a uniform stream. Used by the
// paper to demonstrate the LFM solver at Re = 20 / 200 / 2000.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct Karman {
    double cyl_cx = 2.0;
    double cyl_cy = 1.0;
    double cyl_R  = 0.1;
    double U_inf  = 1.0;
};

// Mark grid cells inside the cylinder as solid (stair-step approximation).
void setup_karman_cylinder(Grid& g, const Karman& k);

// Initialize u = U_inf everywhere (warm start so the wake develops
// naturally rather than impulsively from rest).
void set_uniform_inflow(Grid& g, double U_inf);

// Add a 1 % sinusoidal v-perturbation in a thin band immediately
// behind the cylinder. Breaks the y-symmetry so the wake locks into
// the alternating shedding mode in 2D (where 3D's spanwise modes are
// absent).
void seed_wake_perturbation(Grid& g, const Karman& k, double amplitude = 0.01);

// Build the BC stack: inflow on x=0, outflow on x=Lx, free-slip on
// y=0 / y=Ly, no-slip on the cylinder.
bc::BoundaryManager karman_bcs(double U_inf);

}  // namespace scenarios

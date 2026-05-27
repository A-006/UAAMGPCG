#pragma once
#include "core/grid_3d.h"
#include "scalar/scalar_field_3d.h"

// ──────────────────────────────────────────────────────────────────
// Fire ball / hot plume initial condition (paper Fig. 4).
//
// Seeds a Gaussian hot region of temperature T_hot at the given center;
// the surrounding fluid is at T_ref. Combined with the buoyancy force
// (apply_buoyancy in scalar/scalar_field_3d.h) this produces a rising,
// vortically-tearing thermal plume.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct FireBall {
    std::array<double, 3> center{ {0.5, 0.2, 0.5} };
    double radius = 0.08;
    double T_hot  = 1.0;
    double T_ref  = 0.0;
};

void seed_fire_ball(ScalarField3D& T, const Grid3D& g, const FireBall& fb);

}  // namespace scenarios

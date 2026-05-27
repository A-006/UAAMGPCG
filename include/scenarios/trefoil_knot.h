#pragma once
#include "core/grid_3d.h"

// ──────────────────────────────────────────────────────────────────
// Trefoil knot vortex filament (paper Fig. 7).
//
// Parametrization (standard "(2,3) torus knot" form, scaled to fit
// the unit box centered at `center`):
//
//   x(s) = sin(s) + 2 sin(2s)
//   y(s) = cos(s) - 2 cos(2s)
//   z(s) = -sin(3s)
//
// add_trefoil_knot() discretizes the curve into n_segments and applies
// the same regularized Biot-Savart kernel used by VortexRing, so the
// scenario integrates with ChorinSimulator3D without any extra plumbing.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

struct TrefoilKnot {
    std::array<double, 3> center{ {0.5, 0.5, 0.5} };
    double scale = 0.08;             // bounding box ≈ 6*scale wide
    double core  = 0.025;
    double circulation = 0.4;
    int    n_segments = 360;
};

void add_trefoil_knot(Grid3D& g, const TrefoilKnot& tk);

}  // namespace scenarios

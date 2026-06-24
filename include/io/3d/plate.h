#pragma once
#include "mesh/grid_3d.h"

// ── Flat triangular plate (immersed solid) ──────────────────────────────────
// A generic flat triangular plate, parametrized by chord / span / thickness /
// orientation. Marks every cell inside the (thickened) tilted triangle as
// solid. Used e.g. as a delta-wing geometry, but knows nothing about any
// scenario — it just builds a plate from its parameters.
namespace scenarios {

struct Plate {
    double leading_x = 0.6;  // x of the apex
    double chord     = 0.8;  // root chord length (apex → trailing edge)
    double semi_span = 0.4;  // half of the span (z direction)
    double thickness = 0.02; // half-thickness (y direction)
    double tilt_deg  = 12.0; // orientation: rotation about z
    double y_mid     = 0.5;  // mid-plane y position
};

// Mark grid cells inside the plate as solid.
void setup_plate(Grid3D& g, const Plate& plate);

} // namespace scenarios

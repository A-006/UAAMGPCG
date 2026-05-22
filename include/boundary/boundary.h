#pragma once
#include "core/grid.h"

// Boundary condition enforcement for domain edges and solid obstacles.
// Corresponds to OpenFOAM's fvPatchField — constrains the solution at boundaries.
class BoundaryConditions {
public:
    // Karman vortex street: left inflow U∞, right outflow ∂/∂x=0, top/bottom slip walls
    static void applyKarman(Grid& g, double U_inf);

    // Smoke: four walls no-slip
    static void applySmoke(Grid& g);

    // No-slip on solid-adjacent faces
    static void applySolid(Grid& g);

    // Mark cells within radius R of (cx,cy) as solid
    static void setupCylinder(Grid& g, double cx, double cy, double R);
};

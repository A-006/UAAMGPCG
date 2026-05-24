#pragma once
#include "core/grid.h"

// Advection schemes for the convection term (u·∇)u.
class AdvectionScheme {
public:
    // ── Semi-Lagrangian (original) ──
    static double sampleU(const Grid& g, double x, double y);
    static double sampleV(const Grid& g, double x, double y);
    static void backtrace(const Grid& g, double x, double y, double dt,
                          double& xp, double& yp);
    static void advect(const Grid& oldGrid, Grid& newGrid, double dt);

    // ── Eulerian flux-based — matches icoFoam div(phi,U) Gauss linear ──
    // Non-conservative form, 2nd-order central differences, linear interpolation
    // for face velocities. Updates grid in-place.
    static void advectEulerian(Grid& g, double dt);
};

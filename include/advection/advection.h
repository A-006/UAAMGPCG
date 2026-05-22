#pragma once
#include "core/grid.h"

// Semi-Lagrangian advection scheme (RK2 backtrace + bilinear interpolation).
// Corresponds to OpenFOAM's divScheme — discretization of the convection term (u·∇)u.
class AdvectionScheme {
public:
    // Bilinear interpolation of u-velocity at arbitrary (x,y)
    static double sampleU(const Grid& g, double x, double y);

    // Bilinear interpolation of v-velocity at arbitrary (x,y)
    static double sampleV(const Grid& g, double x, double y);

    // RK2 backtrace from (x,y) backward along velocity field by dt
    static void backtrace(const Grid& g, double x, double y, double dt,
                          double& xp, double& yp);

    // Semi-Lagrangian advection: trace using oldGrid velocities, write to newGrid
    static void advect(const Grid& oldGrid, Grid& newGrid, double dt);
};

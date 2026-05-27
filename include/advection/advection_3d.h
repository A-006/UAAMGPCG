#pragma once
#include "core/grid_3d.h"

// 3D semi-Lagrangian advection — parallels include/advection/advection.h.
//
// RK2 backtrace + trilinear MAC-aware interpolation. Solid cells short-circuit
// to zero. Designed to be used by ChorinSimulator3D and as the velocity-field
// advection step in a future 3D LFM.
class AdvectionScheme3D {
public:
    static double sampleU(const Grid3D& g, double x, double y, double z);
    static double sampleV(const Grid3D& g, double x, double y, double z);
    static double sampleW(const Grid3D& g, double x, double y, double z);

    // RK2 midpoint backtrace from physical position (x,y,z) by dt.
    static void backtrace(const Grid3D& g, double x, double y, double z, double dt,
                          double& xp, double& yp, double& zp);

    // Semi-Lagrangian advection of all three components in old→new.
    static void advect(const Grid3D& oldGrid, Grid3D& newGrid, double dt);
};

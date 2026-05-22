#pragma once
#include "lfm/grid.h"

inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline double divergence(const Grid& g, int i, int j) {
    return (g.u_at(i,j) - g.u_at(i-1,j)) / g.dx
         + (g.v_at(i,j) - g.v_at(i,j-1)) / g.dy;
}

// Bilinear interpolation of velocity fields at arbitrary (x,y)
double sample_u(const Grid& g, double x, double y);
double sample_v(const Grid& g, double x, double y);

// RK2 backtrace: trace from (x,y) backward along velocity field by dt
void backtrace(const Grid& g, double x, double y, double dt,
               double& xp, double& yp);

// Semi-Lagrangian advection: trace using g_old velocities, write to g_new
void advect_velocity(const Grid& g_old, Grid& g_new, double dt);

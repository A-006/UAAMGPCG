#pragma once
#include "core/grid.h"
#include "boundary/boundary.h"
#include <cmath>

/// Set uniform velocity: u = U, v = V on all fluid faces.
/// Applies no-slip on solid cells.
inline void set_uniform(Grid& g, double U, double V) {
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i,j) && !g.is_solid(i+1,j))
                g.u_at(i,j) = U;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++)
            if (!g.is_solid(i,j) && !g.is_solid(i,j+1))
                g.v_at(i,j) = V;
    BoundaryConditions::applySolid(g);
}

/// Set linear shear: u = alpha * y, v = 0.
/// Face positions: u-face at y = (j-0.5)*dy, v-face at y = j*dy.
inline void set_shear(Grid& g, double alpha) {
    double dy = g.dy;
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i,j) && !g.is_solid(i+1,j))
                g.u_at(i,j) = alpha * (j - 0.5) * dy;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++)
            if (!g.is_solid(i,j) && !g.is_solid(i,j+1))
                g.v_at(i,j) = 0.0;
    BoundaryConditions::applySolid(g);
}

/// Set a Lamb-Oseen vortex centered at (cx, cy) with given circulation strength Gamma
/// and core radius r0. Background flow is uniform (U_inf, 0).
/// Velocity: u_theta(r) = Gamma/(2*pi*r) * (1 - exp(-r^2/r0^2))
inline void set_vortex(Grid& g, double cx, double cy, double Gamma, double r0, double U_inf = 1.0) {
    double dx = g.dx, dy = g.dy;
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i+1,j)) continue;
            double x = i * dx;
            double y = (j - 0.5) * dy;
            double rx = x - cx, ry = y - cy;
            double r = std::sqrt(rx*rx + ry*ry);
            if (r < 1e-12) { g.u_at(i,j) = U_inf; continue; }
            double u_theta = Gamma / (2.0 * M_PI * r) * (1.0 - std::exp(-r*r / (r0*r0)));
            g.u_at(i,j) = U_inf - u_theta * ry / r;
        }
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i,j+1)) continue;
            double x = (i - 0.5) * dx;
            double y = j * dy;
            double rx = x - cx, ry = y - cy;
            double r = std::sqrt(rx*rx + ry*ry);
            if (r < 1e-12) { g.v_at(i,j) = 0.0; continue; }
            double u_theta = Gamma / (2.0 * M_PI * r) * (1.0 - std::exp(-r*r / (r0*r0)));
            g.v_at(i,j) = u_theta * rx / r;
        }
    BoundaryConditions::applySolid(g);
}

/// Set Taylor-Green vortex array: u =  sin(kx*x)*cos(ky*y), v = -cos(kx*x)*sin(ky*y)
/// Automatically satisfies div·u = 0 for kx = ky.
inline void set_taylor_green(Grid& g, double kx, double ky) {
    double dx = g.dx, dy = g.dy;
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i+1,j)) continue;
            double x = i * dx;
            double y = (j - 0.5) * dy;
            g.u_at(i,j) = std::sin(kx * x) * std::cos(ky * y);
        }
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i,j+1)) continue;
            double x = (i - 0.5) * dx;
            double y = j * dy;
            g.v_at(i,j) = -std::cos(kx * x) * std::sin(ky * y);
        }
    BoundaryConditions::applySolid(g);
}

/// Compute total circulation: sum of vorticity * cell area over fluid cells
inline double compute_circulation(const Grid& g) {
    double circ = 0.0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j)) continue;
            double w = (g.v_at(i,j) - g.v_at(i-1,j)) / g.dx
                     - (g.u_at(i,j) - g.u_at(i,j-1)) / g.dy;
            circ += w * g.dx * g.dy;
        }
    return circ;
}

/// Compute max vorticity magnitude
inline double max_vorticity(const Grid& g) {
    double max_w = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j)) continue;
            double w = (g.v_at(i,j) - g.v_at(i-1,j)) / g.dx
                     - (g.u_at(i,j) - g.u_at(i,j-1)) / g.dy;
            max_w = std::max(max_w, std::abs(w));
        }
    return max_w;
}

/// Find vortex center: position of max |vorticity|
inline std::pair<double,double> vortex_center(const Grid& g) {
    double max_w = 0, cx = 0, cy = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j)) continue;
            double w = (g.v_at(i,j) - g.v_at(i-1,j)) / g.dx
                     - (g.u_at(i,j) - g.u_at(i,j-1)) / g.dy;
            if (std::abs(w) > max_w) {
                max_w = std::abs(w);
                cx = (i - 0.5) * g.dx;
                cy = (j - 0.5) * g.dy;
            }
        }
    return {cx, cy};
}

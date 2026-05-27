#pragma once
#include "core/grid.h"

// ──────────────────────────────────────────────────────────────────
// Finite-volume calculus (OpenFOAM-style fvc:: namespace).
//
// Operators that act on Grid fields. All work in MAC-grid layout:
//   u-face at (i·dx,  (j-0.5)·dy)
//   v-face at ((i-0.5)·dx,  j·dy)
//   p / scalar at cell center ((i-0.5)·dx, (j-0.5)·dy)
//
// Per-cell scalar variants are inline for use in tight loops. Whole-field
// variants will be added when Field<T> lands.
// ──────────────────────────────────────────────────────────────────
namespace fvc {

// Cell-centered divergence: ∇·u = ∂u/∂x + ∂v/∂y.
inline double divergence(const Grid& g, int i, int j) {
    return (g.u_at(i, j) - g.u_at(i - 1, j)) / g.dx
         + (g.v_at(i, j) - g.v_at(i, j - 1)) / g.dy;
}

// Cell-centered Laplacian of a scalar field s indexed by g.ip(i,j).
// Five-point stencil with uniform spacing.
inline double laplacian(const Grid& g, const std::vector<double>& s, int i, int j) {
    double idx2 = 1.0 / (g.dx * g.dx);
    double idy2 = 1.0 / (g.dy * g.dy);
    return (s[g.ip(i + 1, j)] - 2.0 * s[g.ip(i, j)] + s[g.ip(i - 1, j)]) * idx2
         + (s[g.ip(i, j + 1)] - 2.0 * s[g.ip(i, j)] + s[g.ip(i, j - 1)]) * idy2;
}

// Cell-centered velocity-magnitude squared at cell (i,j).
inline double kinetic_energy(const Grid& g, int i, int j) {
    double uc = 0.5 * (g.u_at(i, j) + g.u_at(i - 1, j));
    double vc = 0.5 * (g.v_at(i, j) + g.v_at(i, j - 1));
    return 0.5 * (uc * uc + vc * vc);
}

// Cell-centered vorticity (2D): ω = ∂v/∂x − ∂u/∂y.
// Uses centered differences across cells; clamps at boundaries.
inline double vorticity(const Grid& g, int i, int j) {
    int ip = std::min(i + 1, g.nx);
    int im = std::max(i - 1, 1);
    int jp = std::min(j + 1, g.ny);
    int jm = std::max(j - 1, 1);
    double dvdx = (g.v_at(ip, j) - g.v_at(im, j)) / (2.0 * g.dx);
    double dudy = (g.u_at(i, jp) - g.u_at(i, jm)) / (2.0 * g.dy);
    return dvdx - dudy;
}

}  // namespace fvc

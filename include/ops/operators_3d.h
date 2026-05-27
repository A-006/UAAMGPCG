#pragma once
#include "core/grid_3d.h"
#include <array>
#include <cmath>

// ──────────────────────────────────────────────────────────────────
// 3D finite-volume calculus — parallels include/ops/operators.h.
//
// All operators are header-only and inline so tight loops pay no
// call overhead. Conventions match the 2D versions:
//   divergence at cell center
//   laplacian on a scalar field indexed by g.ip(i,j,k)
//   vorticity returns a 3-vector (curl of velocity)
// ──────────────────────────────────────────────────────────────────
namespace fvc {

inline double divergence(const Grid3D& g, int i, int j, int k) {
    return (g.u_at(i, j, k) - g.u_at(i - 1, j, k)) / g.dx
         + (g.v_at(i, j, k) - g.v_at(i, j - 1, k)) / g.dy
         + (g.w_at(i, j, k) - g.w_at(i, j, k - 1)) / g.dz;
}

inline double laplacian(const Grid3D& g, const std::vector<double>& s,
                        int i, int j, int k) {
    double idx2 = 1.0 / (g.dx * g.dx);
    double idy2 = 1.0 / (g.dy * g.dy);
    double idz2 = 1.0 / (g.dz * g.dz);
    return (s[g.ip(i + 1, j, k)] - 2.0 * s[g.ip(i, j, k)] + s[g.ip(i - 1, j, k)]) * idx2
         + (s[g.ip(i, j + 1, k)] - 2.0 * s[g.ip(i, j, k)] + s[g.ip(i, j - 1, k)]) * idy2
         + (s[g.ip(i, j, k + 1)] - 2.0 * s[g.ip(i, j, k)] + s[g.ip(i, j, k - 1)]) * idz2;
}

inline double kinetic_energy(const Grid3D& g, int i, int j, int k) {
    double uc = 0.5 * (g.u_at(i, j, k)   + g.u_at(i - 1, j, k));
    double vc = 0.5 * (g.v_at(i, j, k)   + g.v_at(i, j - 1, k));
    double wc = 0.5 * (g.w_at(i, j, k)   + g.w_at(i, j, k - 1));
    return 0.5 * (uc * uc + vc * vc + wc * wc);
}

// 3D vorticity vector ω = ∇ × u  =  (∂w/∂y − ∂v/∂z,
//                                   ∂u/∂z − ∂w/∂x,
//                                   ∂v/∂x − ∂u/∂y).
// Uses centered differences across cells with simple boundary clamps.
inline std::array<double, 3> vorticity(const Grid3D& g, int i, int j, int k) {
    int ip = std::min(i + 1, g.nx), im = std::max(i - 1, 1);
    int jp = std::min(j + 1, g.ny), jm = std::max(j - 1, 1);
    int kp = std::min(k + 1, g.nz), km = std::max(k - 1, 1);
    double dw_dy = (g.w_at(i, jp, k) - g.w_at(i, jm, k)) / (2.0 * g.dy);
    double dv_dz = (g.v_at(i, j, kp) - g.v_at(i, j, km)) / (2.0 * g.dz);
    double du_dz = (g.u_at(i, j, kp) - g.u_at(i, j, km)) / (2.0 * g.dz);
    double dw_dx = (g.w_at(ip, j, k) - g.w_at(im, j, k)) / (2.0 * g.dx);
    double dv_dx = (g.v_at(ip, j, k) - g.v_at(im, j, k)) / (2.0 * g.dx);
    double du_dy = (g.u_at(i, jp, k) - g.u_at(i, jm, k)) / (2.0 * g.dy);
    return { dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy };
}

inline double vorticity_magnitude(const Grid3D& g, int i, int j, int k) {
    auto w = vorticity(g, i, j, k);
    return std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
}

}  // namespace fvc

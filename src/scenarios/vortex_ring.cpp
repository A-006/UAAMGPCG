#include "scenarios/vortex_ring.h"
#include <cmath>

namespace scenarios {

namespace {

// Build an orthonormal pair (e1, e2) perpendicular to `axis`. Used to
// parametrize the ring filament in the plane normal to `axis`.
void make_basis(const std::array<double, 3>& axis,
                std::array<double, 3>& e1,
                std::array<double, 3>& e2) {
    std::array<double, 3> n = axis;
    double m = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (m < 1e-12) m = 1.0;
    n[0] /= m; n[1] /= m; n[2] /= m;

    // Pick a vector not parallel to n
    std::array<double, 3> tmp = (std::abs(n[0]) < 0.9) ? std::array<double,3>{1,0,0}
                                                       : std::array<double,3>{0,1,0};
    // e1 = (tmp × n) normalized
    e1[0] = tmp[1] * n[2] - tmp[2] * n[1];
    e1[1] = tmp[2] * n[0] - tmp[0] * n[2];
    e1[2] = tmp[0] * n[1] - tmp[1] * n[0];
    m = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    e1[0] /= m; e1[1] /= m; e1[2] /= m;
    // e2 = n × e1
    e2[0] = n[1] * e1[2] - n[2] * e1[1];
    e2[1] = n[2] * e1[0] - n[0] * e1[2];
    e2[2] = n[0] * e1[1] - n[1] * e1[0];
}

// Regularized Biot-Savart kernel for a thin filament:
//   u(x) = (Γ/4π) Σ (dl × r) / (|r|² + a²)^{3/2},  r = x − filament_point.
// The a² regularization is a standard Rosenhead/Lamb-Oseen-style fix that
// avoids the 1/r² singularity at the filament and gives a smooth Gaussian-
// like core of radius ~core_radius. Sum is per filament segment.
std::array<double, 3> biot_savart_at(const VortexRing& vr,
                                      const std::array<double, 3>& e1,
                                      const std::array<double, 3>& e2,
                                      double px, double py, double pz) {
    double a2 = vr.core * vr.core;
    double dphi = 2.0 * M_PI / vr.n_segments;
    double Gamma_over_4pi = vr.circulation / (4.0 * M_PI);
    double ux = 0, uy = 0, uz = 0;
    for (int s = 0; s < vr.n_segments; s++) {
        double phi = (s + 0.5) * dphi;
        double cs = std::cos(phi), sn = std::sin(phi);
        // Filament point
        double fx = vr.center[0] + vr.radius * (cs * e1[0] + sn * e2[0]);
        double fy = vr.center[1] + vr.radius * (cs * e1[1] + sn * e2[1]);
        double fz = vr.center[2] + vr.radius * (cs * e1[2] + sn * e2[2]);
        // Tangent dl = R dphi * (-sin·e1 + cos·e2)
        double dlx = vr.radius * dphi * (-sn * e1[0] + cs * e2[0]);
        double dly = vr.radius * dphi * (-sn * e1[1] + cs * e2[1]);
        double dlz = vr.radius * dphi * (-sn * e1[2] + cs * e2[2]);
        double rx = px - fx, ry = py - fy, rz = pz - fz;
        double r2 = rx*rx + ry*ry + rz*rz + a2;
        double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
        // dl × r
        ux += (dly * rz - dlz * ry) * inv_r3;
        uy += (dlz * rx - dlx * rz) * inv_r3;
        uz += (dlx * ry - dly * rx) * inv_r3;
    }
    return { Gamma_over_4pi * ux, Gamma_over_4pi * uy, Gamma_over_4pi * uz };
}

}  // namespace

void add_vortex_ring(Grid3D& g, const VortexRing& vr) {
    std::array<double, 3> e1{}, e2{};
    make_basis(vr.axis, e1, e2);

    // u-face: physical position (i*dx, (j-0.5)*dy, (k-0.5)*dz)
    for (int k = 1; k <= g.nz; k++) {
        for (int j = 1; j <= g.ny; j++) {
            for (int i = 0; i <= g.nx; i++) {
                double px = i * g.dx;
                double py = (j - 0.5) * g.dy;
                double pz = (k - 0.5) * g.dz;
                auto u = biot_savart_at(vr, e1, e2, px, py, pz);
                g.u_at(i, j, k) += u[0];
            }
        }
    }
    // v-face: ((i-0.5)*dx, j*dy, (k-0.5)*dz)
    for (int k = 1; k <= g.nz; k++) {
        for (int j = 0; j <= g.ny; j++) {
            for (int i = 1; i <= g.nx; i++) {
                double px = (i - 0.5) * g.dx;
                double py = j * g.dy;
                double pz = (k - 0.5) * g.dz;
                auto u = biot_savart_at(vr, e1, e2, px, py, pz);
                g.v_at(i, j, k) += u[1];
            }
        }
    }
    // w-face: ((i-0.5)*dx, (j-0.5)*dy, k*dz)
    for (int k = 0; k <= g.nz; k++) {
        for (int j = 1; j <= g.ny; j++) {
            for (int i = 1; i <= g.nx; i++) {
                double px = (i - 0.5) * g.dx;
                double py = (j - 0.5) * g.dy;
                double pz = k * g.dz;
                auto u = biot_savart_at(vr, e1, e2, px, py, pz);
                g.w_at(i, j, k) += u[2];
            }
        }
    }
}

}  // namespace scenarios

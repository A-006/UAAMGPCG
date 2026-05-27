#include "scenarios/trefoil_knot.h"
#include <cmath>

namespace scenarios {

namespace {

struct Point { double x, y, z; };

// Parametrize the trefoil curve and pre-build the filament as a closed
// polyline (segment endpoints in physical units).
std::vector<Point> build_curve(const TrefoilKnot& tk) {
    std::vector<Point> pts(tk.n_segments + 1);
    for (int i = 0; i <= tk.n_segments; i++) {
        double s = 2.0 * M_PI * i / tk.n_segments;
        double x = std::sin(s) + 2.0 * std::sin(2.0 * s);
        double y = std::cos(s) - 2.0 * std::cos(2.0 * s);
        double z = -std::sin(3.0 * s);
        pts[i] = { tk.center[0] + tk.scale * x,
                   tk.center[1] + tk.scale * y,
                   tk.center[2] + tk.scale * z };
    }
    return pts;
}

// Velocity from a single straight segment via regularized Biot-Savart:
//   contribution = (Γ/4π) · (dl × r) / (|r|² + a²)^{3/2}
std::array<double, 3> segment_velocity(const Point& p, const Point& a, const Point& b,
                                        double Gamma_over_4pi, double core2) {
    double dlx = b.x - a.x, dly = b.y - a.y, dlz = b.z - a.z;
    double mx = 0.5 * (a.x + b.x), my = 0.5 * (a.y + b.y), mz = 0.5 * (a.z + b.z);
    double rx = p.x - mx, ry = p.y - my, rz = p.z - mz;
    double r2 = rx*rx + ry*ry + rz*rz + core2;
    double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
    return {
        Gamma_over_4pi * (dly * rz - dlz * ry) * inv_r3,
        Gamma_over_4pi * (dlz * rx - dlx * rz) * inv_r3,
        Gamma_over_4pi * (dlx * ry - dly * rx) * inv_r3,
    };
}

std::array<double, 3> velocity_at(const std::vector<Point>& pts, Point p,
                                   double Gamma, double core) {
    double c4pi = Gamma / (4.0 * M_PI);
    double c2 = core * core;
    double ux = 0, uy = 0, uz = 0;
    for (size_t i = 0; i + 1 < pts.size(); i++) {
        auto v = segment_velocity(p, pts[i], pts[i + 1], c4pi, c2);
        ux += v[0]; uy += v[1]; uz += v[2];
    }
    return { ux, uy, uz };
}

}  // namespace

void add_trefoil_knot(Grid3D& g, const TrefoilKnot& tk) {
    auto pts = build_curve(tk);

    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++) {
                Point p = { i * g.dx, (j - 0.5) * g.dy, (k - 0.5) * g.dz };
                auto u = velocity_at(pts, p, tk.circulation, tk.core);
                g.u_at(i, j, k) += u[0];
            }
    for (int k = 1; k <= g.nz; k++)
        for (int j = 0; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                Point p = { (i - 0.5) * g.dx, j * g.dy, (k - 0.5) * g.dz };
                auto u = velocity_at(pts, p, tk.circulation, tk.core);
                g.v_at(i, j, k) += u[1];
            }
    for (int k = 0; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                Point p = { (i - 0.5) * g.dx, (j - 0.5) * g.dy, k * g.dz };
                auto u = velocity_at(pts, p, tk.circulation, tk.core);
                g.w_at(i, j, k) += u[2];
            }
}

}  // namespace scenarios

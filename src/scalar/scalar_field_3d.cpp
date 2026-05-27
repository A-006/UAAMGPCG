#include "scalar/scalar_field_3d.h"
#include <algorithm>

double ScalarField3D::sample(const ScalarField3D& s, const Grid3D& g,
                              double x, double y, double z) {
    x = Mesh3D::clamp(x, 0.0, g.Lx());
    y = Mesh3D::clamp(y, 0.0, g.Ly());
    z = Mesh3D::clamp(z, 0.0, g.Lz());
    // Cell centers at ((i-0.5)dx, (j-0.5)dy, (k-0.5)dz)
    double fi = x / g.dx + 0.5;
    double fj = y / g.dy + 0.5;
    double fk = z / g.dz + 0.5;
    int i0 = Mesh3D::clamp((int)fi, 0, g.nx + 1);
    int j0 = Mesh3D::clamp((int)fj, 0, g.ny + 1);
    int k0 = Mesh3D::clamp((int)fk, 0, g.nz + 1);
    int i1 = std::min(i0 + 1, g.nx + 1);
    int j1 = std::min(j0 + 1, g.ny + 1);
    int k1 = std::min(k0 + 1, g.nz + 1);
    double wi = (i1 == i0) ? 0.0 : (fi - i0);
    double wj = (j1 == j0) ? 0.0 : (fj - j0);
    double wk = (k1 == k0) ? 0.0 : (fk - k0);
    auto V = [&](int a, int b, int c) { return s.data()[g.ip(a, b, c)]; };
    double c000 = V(i0, j0, k0), c100 = V(i1, j0, k0);
    double c010 = V(i0, j1, k0), c110 = V(i1, j1, k0);
    double c001 = V(i0, j0, k1), c101 = V(i1, j0, k1);
    double c011 = V(i0, j1, k1), c111 = V(i1, j1, k1);
    double c00 = c000 * (1 - wi) + c100 * wi;
    double c10 = c010 * (1 - wi) + c110 * wi;
    double c01 = c001 * (1 - wi) + c101 * wi;
    double c11 = c011 * (1 - wi) + c111 * wi;
    double c0 = c00 * (1 - wj) + c10 * wj;
    double c1 = c01 * (1 - wj) + c11 * wj;
    return c0 * (1 - wk) + c1 * wk;
}

void ScalarField3D::advect(const ScalarField3D& src, ScalarField3D& dst,
                            const Grid3D& flow, double dt) {
    for (int k = 1; k <= flow.nz; k++)
        for (int j = 1; j <= flow.ny; j++)
            for (int i = 1; i <= flow.nx; i++) {
                if (flow.is_solid(i, j, k)) { dst(i, j, k) = 0.0; continue; }
                double x = (i - 0.5) * flow.dx;
                double y = (j - 0.5) * flow.dy;
                double z = (k - 0.5) * flow.dz;
                double xp, yp, zp;
                AdvectionScheme3D::backtrace(flow, x, y, z, dt, xp, yp, zp);
                dst(i, j, k) = sample(src, flow, xp, yp, zp);
            }
}

void apply_buoyancy(Grid3D& g, const ScalarField3D& T, double T_ref,
                    double beta, double dt) {
    // w-face at (i-0.5)dx, (j-0.5)dy, k*dz  is between cells (i,j,k) and (i,j,k+1)
    for (int k = 1; k < g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                if (g.is_solid(i, j, k) || g.is_solid(i, j, k + 1)) continue;
                double T_face = 0.5 * (T(i, j, k) + T(i, j, k + 1));
                g.w_at(i, j, k) += dt * beta * (T_face - T_ref);
            }
}

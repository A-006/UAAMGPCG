#include "advection/advection_3d.h"
#include <algorithm>

namespace {

// Trilinear interpolation helper. Given fractional positions (fi, fj, fk)
// in a MAC layout with size sx × sy × sz and an indexer flat(i,j,k),
// sample the field at the eight surrounding grid points.
template <typename Idx, typename Field>
double trilinear(const Field& f, double fi, double fj, double fk,
                 int sx, int sy, int sz, Idx flat) {
    int i0 = Mesh3D::clamp((int)fi, 0, sx - 1);
    int j0 = Mesh3D::clamp((int)fj, 0, sy - 1);
    int k0 = Mesh3D::clamp((int)fk, 0, sz - 1);
    int i1 = std::min(i0 + 1, sx - 1);
    int j1 = std::min(j0 + 1, sy - 1);
    int k1 = std::min(k0 + 1, sz - 1);
    double wi = (i1 == i0) ? 0.0 : (fi - i0);
    double wj = (j1 == j0) ? 0.0 : (fj - j0);
    double wk = (k1 == k0) ? 0.0 : (fk - k0);
    double c000 = f[flat(i0, j0, k0)];
    double c100 = f[flat(i1, j0, k0)];
    double c010 = f[flat(i0, j1, k0)];
    double c110 = f[flat(i1, j1, k0)];
    double c001 = f[flat(i0, j0, k1)];
    double c101 = f[flat(i1, j0, k1)];
    double c011 = f[flat(i0, j1, k1)];
    double c111 = f[flat(i1, j1, k1)];
    double c00 = c000 * (1 - wi) + c100 * wi;
    double c10 = c010 * (1 - wi) + c110 * wi;
    double c01 = c001 * (1 - wi) + c101 * wi;
    double c11 = c011 * (1 - wi) + c111 * wi;
    double c0 = c00 * (1 - wj) + c10 * wj;
    double c1 = c01 * (1 - wj) + c11 * wj;
    return c0 * (1 - wk) + c1 * wk;
}

}  // namespace

double AdvectionScheme3D::sampleU(const Grid3D& g, double x, double y, double z) {
    x = Mesh3D::clamp(x, 0.0, g.Lx());
    y = Mesh3D::clamp(y, 0.0, g.Ly());
    z = Mesh3D::clamp(z, 0.0, g.Lz());
    // u-face: (i*dx, (j-0.5)*dy, (k-0.5)*dz), i∈[0,nx], j∈[1,ny], k∈[1,nz]
    double fi = x / g.dx;
    double fj = y / g.dy + 0.5;
    double fk = z / g.dz + 0.5;
    auto idx = [&](int i, int j, int k) { return g.iu(i, j, k); };
    return trilinear(g.u, fi, fj, fk, g.nx + 1, g.ny + 2, g.nz + 2, idx);
}

double AdvectionScheme3D::sampleV(const Grid3D& g, double x, double y, double z) {
    x = Mesh3D::clamp(x, 0.0, g.Lx());
    y = Mesh3D::clamp(y, 0.0, g.Ly());
    z = Mesh3D::clamp(z, 0.0, g.Lz());
    double fi = x / g.dx + 0.5;
    double fj = y / g.dy;
    double fk = z / g.dz + 0.5;
    auto idx = [&](int i, int j, int k) { return g.iv(i, j, k); };
    return trilinear(g.v, fi, fj, fk, g.nx + 2, g.ny + 1, g.nz + 2, idx);
}

double AdvectionScheme3D::sampleW(const Grid3D& g, double x, double y, double z) {
    x = Mesh3D::clamp(x, 0.0, g.Lx());
    y = Mesh3D::clamp(y, 0.0, g.Ly());
    z = Mesh3D::clamp(z, 0.0, g.Lz());
    double fi = x / g.dx + 0.5;
    double fj = y / g.dy + 0.5;
    double fk = z / g.dz;
    auto idx = [&](int i, int j, int k) { return g.iw(i, j, k); };
    return trilinear(g.w, fi, fj, fk, g.nx + 2, g.ny + 2, g.nz + 1, idx);
}

void AdvectionScheme3D::backtrace(const Grid3D& g, double x, double y, double z,
                                   double dt,
                                   double& xp, double& yp, double& zp) {
    double u0 = sampleU(g, x, y, z);
    double v0 = sampleV(g, x, y, z);
    double w0 = sampleW(g, x, y, z);
    double xm = x - 0.5 * dt * u0;
    double ym = y - 0.5 * dt * v0;
    double zm = z - 0.5 * dt * w0;
    double um = sampleU(g, xm, ym, zm);
    double vm = sampleV(g, xm, ym, zm);
    double wm = sampleW(g, xm, ym, zm);
    xp = x - dt * um;
    yp = y - dt * vm;
    zp = z - dt * wm;
}

void AdvectionScheme3D::advect(const Grid3D& oldGrid, Grid3D& newGrid, double dt) {
    int nx = oldGrid.nx, ny = oldGrid.ny, nz = oldGrid.nz;

    // u-faces
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i < nx; i++) {
                if (oldGrid.is_solid(i, j, k) || oldGrid.is_solid(i + 1, j, k)) {
                    newGrid.u_at(i, j, k) = 0.0;
                    continue;
                }
                double xf = i * oldGrid.dx;
                double yf = (j - 0.5) * oldGrid.dy;
                double zf = (k - 0.5) * oldGrid.dz;
                double xp, yp, zp;
                backtrace(oldGrid, xf, yf, zf, dt, xp, yp, zp);
                newGrid.u_at(i, j, k) = sampleU(oldGrid, xp, yp, zp);
            }

    // v-faces
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j < ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (oldGrid.is_solid(i, j, k) || oldGrid.is_solid(i, j + 1, k)) {
                    newGrid.v_at(i, j, k) = 0.0;
                    continue;
                }
                double xf = (i - 0.5) * oldGrid.dx;
                double yf = j * oldGrid.dy;
                double zf = (k - 0.5) * oldGrid.dz;
                double xp, yp, zp;
                backtrace(oldGrid, xf, yf, zf, dt, xp, yp, zp);
                newGrid.v_at(i, j, k) = sampleV(oldGrid, xp, yp, zp);
            }

    // w-faces
    for (int k = 1; k < nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (oldGrid.is_solid(i, j, k) || oldGrid.is_solid(i, j, k + 1)) {
                    newGrid.w_at(i, j, k) = 0.0;
                    continue;
                }
                double xf = (i - 0.5) * oldGrid.dx;
                double yf = (j - 0.5) * oldGrid.dy;
                double zf = k * oldGrid.dz;
                double xp, yp, zp;
                backtrace(oldGrid, xf, yf, zf, dt, xp, yp, zp);
                newGrid.w_at(i, j, k) = sampleW(oldGrid, xp, yp, zp);
            }
}

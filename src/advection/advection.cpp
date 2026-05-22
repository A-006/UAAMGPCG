#include "advection/advection.h"
#include <algorithm>

double AdvectionScheme::sampleU(const Grid& g, double x, double y) {
    x = Grid::clamp(x, 0.0, g.nx * g.dx);
    y = Grid::clamp(y, 0.0, g.ny * g.dy);

    double fi = x / g.dx;
    double fj = y / g.dy + 0.5;

    int i0 = (int)fi;
    int j0 = (int)fj;
    i0 = Grid::clamp(i0, 0, g.nx);
    j0 = Grid::clamp(j0, 0, g.ny+1);

    int i1 = std::min(i0+1, g.nx);
    int j1 = std::min(j0+1, g.ny+1);

    double wi = fi - i0;
    double wj = fj - j0;
    if (i1 == i0) wi = 0.0;
    if (j1 == j0) wj = 0.0;

    double v00 = g.u_at(i0,j0), v10 = g.u_at(i1,j0);
    double v01 = g.u_at(i0,j1), v11 = g.u_at(i1,j1);

    return (1-wi)*(1-wj)*v00 + wi*(1-wj)*v10 + (1-wi)*wj*v01 + wi*wj*v11;
}

double AdvectionScheme::sampleV(const Grid& g, double x, double y) {
    x = Grid::clamp(x, 0.0, g.nx * g.dx);
    y = Grid::clamp(y, 0.0, g.ny * g.dy);

    double fi = x / g.dx + 0.5;
    double fj = y / g.dy;

    int i0 = (int)fi;
    int j0 = (int)fj;
    i0 = Grid::clamp(i0, 0, g.nx+1);
    j0 = Grid::clamp(j0, 0, g.ny);

    int i1 = std::min(i0+1, g.nx+1);
    int j1 = std::min(j0+1, g.ny);

    double wi = fi - i0;
    double wj = fj - j0;
    if (i1 == i0) wi = 0.0;
    if (j1 == j0) wj = 0.0;

    double v00 = g.v_at(i0,j0), v10 = g.v_at(i1,j0);
    double v01 = g.v_at(i0,j1), v11 = g.v_at(i1,j1);

    return (1-wi)*(1-wj)*v00 + wi*(1-wj)*v10 + (1-wi)*wj*v01 + wi*wj*v11;
}

void AdvectionScheme::backtrace(const Grid& g, double x, double y, double dt,
                                double& xp, double& yp) {
    double u0 = sampleU(g, x, y);
    double v0 = sampleV(g, x, y);
    double xm = x - 0.5 * dt * u0;
    double ym = y - 0.5 * dt * v0;
    double um = sampleU(g, xm, ym);
    double vm = sampleV(g, xm, ym);
    xp = x - dt * um;
    yp = y - dt * vm;
}

void AdvectionScheme::advect(const Grid& oldGrid, Grid& newGrid, double dt) {
    int nx = oldGrid.nx, ny = oldGrid.ny;

    for (int i = 1; i < nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (oldGrid.is_solid(i,j) || oldGrid.is_solid(i+1,j)) {
                newGrid.u_at(i,j) = 0.0;
                continue;
            }
            double xf = i * oldGrid.dx;
            double yf = (j - 0.5) * oldGrid.dy;
            double xp, yp;
            backtrace(oldGrid, xf, yf, dt, xp, yp);
            newGrid.u_at(i,j) = sampleU(oldGrid, xp, yp);
        }
    }
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j < ny; j++) {
            if (oldGrid.is_solid(i,j) || oldGrid.is_solid(i,j+1)) {
                newGrid.v_at(i,j) = 0.0;
                continue;
            }
            double xf = (i - 0.5) * oldGrid.dx;
            double yf = j * oldGrid.dy;
            double xp, yp;
            backtrace(oldGrid, xf, yf, dt, xp, yp);
            newGrid.v_at(i,j) = sampleV(oldGrid, xp, yp);
        }
    }
}

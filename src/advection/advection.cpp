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

// ── FVM conservative flux advection — matches icoFoam div(phi,U) Gauss linear ──
// Each velocity component is updated by summing advective fluxes through
// its control volume faces. Face velocities are linearly interpolated.
void AdvectionScheme::advectEulerian(Grid& g, double dt) {
    int nx = g.nx, ny = g.ny;
    double dx = g.dx, dy = g.dy;
    std::vector<double> u_old = g.u, v_old = g.v;

    // ── Advect u (control volume: [i*dx, (i+1)*dx] × [(j-1)*dy, j*dy]) ──
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i+1,j)) continue;
            double uC = u_old[g.iu(i,j)];

            // === X-fluxes (through left/right faces of u-CV) ===
            // Left face (x = i*dx): flux = u_face * u_advected
            // u_face at left = u(i-1,j), advected u = (u(i-1,j) + uC) / 2
            double flux_L = 0.0;
            if (i > 1 && !g.is_solid(i-1,j) && !g.is_solid(i,j)) {
                double uL = u_old[g.iu(i-1,j)];
                flux_L = 0.5 * (uL + uC) * 0.5 * (uL + uC) * dy;  // u_face * u_interp * dy
            } else if (i == 1) {
                // Inlet: u(0,j) = U_inf
                flux_L = uC * uC * dy;  // u_face=uC, u_advected=uC (inlet BC)
            }
            // Right face (x = (i+1)*dx)
            double flux_R = 0.0;
            if (i < nx-1 && !g.is_solid(i+1,j) && !g.is_solid(i+2,j)) {
                double uR = u_old[g.iu(i+1,j)];
                flux_R = 0.5 * (uC + uR) * 0.5 * (uC + uR) * dy;
            } else if (i == nx-1) {
                // Outlet: zero gradient → u(nx,j) = u(nx-1,j)
                flux_R = uC * uC * dy;
            }

            // === Y-fluxes (through bottom/top faces of u-CV) ===
            // Bottom face: v interpolated to u-CV bottom corner
            double flux_B = 0.0;
            if (j > 1) {
                double vB = 0.0; int cnt=0; double s=0;
                if (!g.is_solid(i,j-1) && !g.is_solid(i,j)) { s+=v_old[g.iv(i,j-1)]; cnt++; }
                if (i<nx && !g.is_solid(i+1,j-1) && !g.is_solid(i+1,j)) { s+=v_old[g.iv(i+1,j-1)]; cnt++; }
                if (cnt>0) vB = s/cnt;
                double uB = u_old[g.iu(i,j-1)];  // u below
                flux_B = vB * 0.5 * (uC + uB) * dx;
            }
            // Top face
            double flux_T = 0.0;
            if (j < ny) {
                double vT = 0.0; int cnt=0; double s=0;
                if (!g.is_solid(i,j) && !g.is_solid(i,j+1)) { s+=v_old[g.iv(i,j)]; cnt++; }
                if (i<nx && !g.is_solid(i+1,j) && !g.is_solid(i+1,j+1)) { s+=v_old[g.iv(i+1,j)]; cnt++; }
                if (cnt>0) vT = s/cnt;
                double uT = u_old[g.iu(i,j+1)];
                flux_T = vT * 0.5 * (uC + uT) * dx;
            }

            double vol = dx * dy;
            g.u_at(i,j) = uC - dt * (flux_R - flux_L + flux_T - flux_B) / vol;
        }
    }

    // ── Advect v (control volume: [(i-1)*dx, i*dx] × [j*dy, (j+1)*dy]) ──
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j < ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i,j+1)) continue;
            double vC = v_old[g.iv(i,j)];

            // === Y-fluxes ===
            double flux_B = 0.0;
            if (j > 1 && !g.is_solid(i,j-1) && !g.is_solid(i,j)) {
                double vB = v_old[g.iv(i,j-1)];
                flux_B = 0.5 * (vB + vC) * 0.5 * (vB + vC) * dx;
            } else if (j == 1) {
                flux_B = vC * vC * dx;  // symmetry/slip
            }
            double flux_T = 0.0;
            if (j < ny-1 && !g.is_solid(i,j+1) && !g.is_solid(i,j+2)) {
                double vT = v_old[g.iv(i,j+1)];
                flux_T = 0.5 * (vC + vT) * 0.5 * (vC + vT) * dx;
            } else if (j == ny-1) {
                flux_T = vC * vC * dx;  // symmetry
            }

            // === X-fluxes ===
            double flux_L = 0.0;
            if (i > 1) {
                double uL = 0.0; int cnt=0; double s=0;
                if (!g.is_solid(i-1,j) && !g.is_solid(i,j)) { s+=u_old[g.iu(i-1,j)]; cnt++; }
                if (j<ny && !g.is_solid(i-1,j+1) && !g.is_solid(i,j+1)) { s+=u_old[g.iu(i-1,j+1)]; cnt++; }
                if (cnt>0) uL = s/cnt;
                double vL = v_old[g.iv(i-1,j)];
                flux_L = uL * 0.5 * (vC + vL) * dy;
            }
            double flux_R = 0.0;
            if (i < nx) {
                double uR = 0.0; int cnt=0; double s=0;
                if (!g.is_solid(i,j) && !g.is_solid(i+1,j)) { s+=u_old[g.iu(i,j)]; cnt++; }
                if (j<ny && !g.is_solid(i,j+1) && !g.is_solid(i+1,j+1)) { s+=u_old[g.iu(i,j+1)]; cnt++; }
                if (cnt>0) uR = s/cnt;
                double vR = v_old[g.iv(i+1,j)];
                flux_R = uR * 0.5 * (vC + vR) * dy;
            }

            double vol = dx * dy;
            g.v_at(i,j) = vC - dt * (flux_R - flux_L + flux_T - flux_B) / vol;
        }
    }
}

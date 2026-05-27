#include "pressure/pressure_3d.h"
#include "ops/operators_3d.h"
#include <algorithm>

void PressureProjection3D::project(Grid3D& g, double dt, Solver3D& solver,
                                    int max_iter, double tol) {
    int nx = g.nx, ny = g.ny, nz = g.nz;

    // 1. Build RHS: ∇·u / Δt
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                if (!g.is_solid(i, j, k))
                    rhs[g.ip(i, j, k)] = fvc::divergence(g, i, j, k) / dt;

    // 2. Solve ∇²p = rhs
    std::fill(g.p.begin(), g.p.end(), 0.0);
    solver.solve(g, rhs, max_iter, tol);

    // 3. Velocity correction: u ← u - Δt · ∇p
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i < nx; i++) {
                if (g.is_solid(i, j, k) || g.is_solid(i + 1, j, k)) continue;
                g.u_at(i, j, k) -= dt * (g.p_at(i + 1, j, k) - g.p_at(i, j, k)) / g.dx;
            }
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j < ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (g.is_solid(i, j, k) || g.is_solid(i, j + 1, k)) continue;
                g.v_at(i, j, k) -= dt * (g.p_at(i, j + 1, k) - g.p_at(i, j, k)) / g.dy;
            }
    for (int k = 1; k < nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (g.is_solid(i, j, k) || g.is_solid(i, j, k + 1)) continue;
                g.w_at(i, j, k) -= dt * (g.p_at(i, j, k + 1) - g.p_at(i, j, k)) / g.dz;
            }
}

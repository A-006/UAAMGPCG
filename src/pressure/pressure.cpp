#include "pressure/pressure.h"
#include <algorithm>

void PressureProjection::project(Grid& g, double dt, PoissonSolver& solver,
                                  int max_iter, double tol) {
    int nx = g.nx, ny = g.ny;

    // 1. Build RHS: rhs = ∇·ũ / Δt
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            if (!g.is_solid(i,j))
                rhs[g.ip(i,j)] = g.divergence(i, j) / dt;

    // 2. Solve ∇²p = rhs
    std::fill(g.p.begin(), g.p.end(), 0.0);
    solver.solve(g, rhs, max_iter, tol);

    // 3. Velocity correction: u ← ũ - Δt·∇p
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i+1,j)) continue;
            g.u_at(i,j) -= dt * (g.p_at(i+1,j) - g.p_at(i,j)) / g.dx;
        }
    }
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j < ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i,j+1)) continue;
            g.v_at(i,j) -= dt * (g.p_at(i,j+1) - g.p_at(i,j)) / g.dy;
        }
    }
}

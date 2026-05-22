#include "lfm/pressure.h"
#include "lfm/advection.h"
#include "lfm/poisson_jacobi.h"
#include <algorithm>

void pressure_projection(Grid& g, double dt, int jacobi_iters) {
    int nx = g.nx, ny = g.ny;

    // 1. Build RHS: rhs = ∇·ũ / Δt
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            if (!g.is_solid(i,j))
                rhs[g.ip(i,j)] = divergence(g, i, j) / dt;

    // 2. Poisson solve
    std::fill(g.p.begin(), g.p.end(), 0.0);
    jacobi_solve(g, rhs, jacobi_iters);

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

#include "lfm/pressure.h"
#include "lfm/advection.h"
#include "lfm/poisson_jacobi.h"
#include "lfm/poisson_rbgs.h"
#include "lfm/poisson_cg.h"
#include "lfm/poisson_pcg.h"
#include <algorithm>

void pressure_projection(Grid& g, double dt, const Config& cfg) {
    int nx = g.nx, ny = g.ny;

    // 1. Build RHS: rhs = ∇·ũ / Δt
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            if (!g.is_solid(i,j))
                rhs[g.ip(i,j)] = divergence(g, i, j) / dt;

    // 2. Solve ∇²p = rhs
    std::fill(g.p.begin(), g.p.end(), 0.0);
    switch (cfg.solver) {
        case Solver::Jacobi:
            jacobi_solve(g, rhs, cfg.solve_iters);
            break;
        case Solver::RBGS:
            rbgs_solve(g, rhs, cfg.solve_iters);
            break;
        case Solver::CG:
            cg_solve(g, rhs, cfg.solve_iters, cfg.solve_tol);
            break;
        case Solver::PCG:
            pcg_solve(g, rhs, cfg.solve_iters, cfg.solve_tol);
            break;
    }

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

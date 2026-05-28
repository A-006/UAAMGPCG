#include "pressure/pressure.h"
#include "ops/operators.h"
#include <algorithm>

void PressureProjection::project(Grid& g, double dt, Solver& solver,
                                  int max_iter, double tol) {
    int nx = g.nx, ny = g.ny;

    // 1. Build RHS: rhs = ∇·ũ / Δt
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            if (!g.is_solid(i,j))
                rhs[g.ip(i,j)] = fvc::divergence(g, i, j) / dt;

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

void PressureProjection::recoverStaticPressure(Grid& g, Solver& solver,
                                               int max_iter, double tol) {
    int nx = g.nx, ny = g.ny;
    double dx = g.dx, dy = g.dy;
    auto uc = [&](int i, int j) {
        i = std::clamp(i, 1, nx); j = std::clamp(j, 1, ny);
        return 0.5 * (g.u_at(i, j) + g.u_at(i - 1, j));
    };
    auto vc = [&](int i, int j) {
        i = std::clamp(i, 1, nx); j = std::clamp(j, 1, ny);
        return 0.5 * (g.v_at(i, j) + g.v_at(i, j - 1));
    };

    // RHS of the PPE (same sign convention as project's ∇²p = rhs).
    // Cross derivatives are dropped next to solids (stair-step protection),
    // matching the velocity-gradient treatment used elsewhere.
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++) {
            if (g.is_solid(i, j)) continue;
            double du_dx = (g.u_at(i, j) - g.u_at(i - 1, j)) / dx;
            double dv_dy = (g.v_at(i, j) - g.v_at(i, j - 1)) / dy;
            bool sL = g.is_solid(i-1,j), sR = g.is_solid(i+1,j);
            bool sB = g.is_solid(i,j-1), sT = g.is_solid(i,j+1);
            double du_dy = (sB || sT) ? 0.0 : (uc(i, j+1) - uc(i, j-1)) / (2*dy);
            double dv_dx = (sL || sR) ? 0.0 : (vc(i+1, j) - vc(i-1, j)) / (2*dx);
            rhs[g.ip(i, j)] = -(du_dx*du_dx + 2.0*du_dy*dv_dx + dv_dy*dv_dy);
        }

    std::fill(g.p.begin(), g.p.end(), 0.0);
    solver.solve(g, rhs, max_iter, tol);
}

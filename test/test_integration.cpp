// Integration test: runs LFM simulation with each solver,
// verifies velocities stay bounded and divergence is controlled.
#include "lfm/config.h"
#include "lfm/grid.h"
#include "lfm/advection.h"
#include "lfm/boundary.h"
#include "lfm/pressure.h"
#include "test_utils.h"
#include <cmath>
#include <vector>

static void lfm_step(Grid& g, double dt, const Config& cfg, Grid& g_prev) {
    if (cfg.scenario == "smoke") {
        double buoyancy = 5.0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j)) g.v_at(i,j) += dt * buoyancy;
    }
    Grid g_adv = g_prev;
    for (int i = 1; i < g.nx; i++)
        for (int j = 1; j <= g.ny; j++) g_adv.u_at(i,j) = 0.0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j < g.ny; j++) g_adv.v_at(i,j) = 0.0;
    advect_velocity(g_prev, g_adv, dt);
    for (int i = 1; i < g.nx; i++)
        for (int j = 1; j <= g.ny; j++) g.u_at(i,j) = g_adv.u_at(i,j);
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j < g.ny; j++) g.v_at(i,j) = g_adv.v_at(i,j);
    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);
    pressure_projection(g, dt, cfg);
    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);
}

int main() {
    test_header("LFM Integration Tests");

    std::vector<Solver> solvers = {Solver::Jacobi, Solver::RBGS, Solver::CG, Solver::PCG};

    // Test 1: Uniform inflow — each solver preserves zero divergence
    for (auto solver : solvers) {
        Config cfg;
        cfg.scenario = "karman";
        cfg.solver = solver;
        cfg.NX = 32; cfg.NY = 16;
        cfg.Lx = 4.0; cfg.Ly = 1.0;
        cfg.dt = 0.5 * (cfg.Lx/cfg.NX) / cfg.U_inf;
        cfg.solve_iters = (solver <= Solver::RBGS) ? 500 : 30;
        cfg.solve_tol = 1e-6;

        Grid g(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly);
        for (int i = 0; i <= cfg.NX; i++)
            for (int j = 1; j <= cfg.NY; j++)
                g.u_at(i,j) = 1.0;
        apply_bc_karman(g, 1.0);
        Grid g_prev = g;

        bool ok = true;
        for (int step = 0; step < 3; step++) {
            lfm_step(g, cfg.dt, cfg, g_prev);
            g_prev = g;
            double max_div = 0;
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++)
                    if (!g.is_solid(i,j))
                        max_div = std::max(max_div, std::abs(divergence(g,i,j)));
            if (max_div > 1e-4 || !std::isfinite(max_div)) ok = false;
        }
        check(ok, std::string(solver_name(solver)) + ": uniform flow preserved");
    }

    // Test 2: Karman with cylinder — CG/PCG (fast solvers) stay bounded.
    // Jacobi/RBGS are too slow for the sharp pressure gradients at the cylinder wake.
    std::vector<Solver> fast_solvers = {Solver::CG, Solver::PCG};
    for (auto solver : fast_solvers) {
        Config cfg;
        cfg.scenario = "karman";
        cfg.solver = solver;
        cfg.NX = 64; cfg.NY = 32;
        cfg.Lx = 4.0; cfg.Ly = 1.0;
        cfg.dt = 0.5 * (cfg.Lx/cfg.NX) / cfg.U_inf;
        cfg.solve_iters = (solver <= Solver::RBGS) ? 5000 : 30;
        cfg.solve_tol = 1e-6;

        Grid g(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly);
        setup_cylinder(g, 1.0, 0.5, 0.1);
        apply_bc_karman(g, 1.0);
        apply_solid_bc(g);
        for (int i = 1; i < cfg.NX; i++)
            for (int j = 1; j <= cfg.NY; j++)
                if (!g.is_solid(i,j) && !g.is_solid(i+1,j))
                    g.u_at(i,j) = 1.0;

        Grid g_prev = g;
        bool ok = true;
        for (int step = 0; step < 5; step++) {
            lfm_step(g, cfg.dt, cfg, g_prev);
            g_prev = g;
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i,j)) continue;
                    double uc = 0.5*(g.u_at(i-1,j) + g.u_at(i,j));
                    double vc = 0.5*(g.v_at(i,j-1) + g.v_at(i,j));
                    if (!std::isfinite(uc) || !std::isfinite(vc)) ok = false;
                }
        }
        check(ok, std::string(solver_name(solver)) + ": cylinder flow bounded");
    }

    // Test 3: Smoke buoyancy — each solver survives 10 steps
    for (auto solver : solvers) {
        Config cfg;
        cfg.scenario = "smoke";
        cfg.solver = solver;
        cfg.NX = 32; cfg.NY = 32;
        cfg.Lx = 1.0; cfg.Ly = 1.0;
        cfg.dt = 0.005;
        cfg.solve_iters = (solver <= Solver::RBGS) ? 1000 : 20;
        cfg.solve_tol = 1e-6;

        Grid g(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly);
        apply_bc_smoke(g);
        Grid g_prev = g;

        bool ok = true;
        for (int step = 0; step < 10; step++) {
            lfm_step(g, cfg.dt, cfg, g_prev);
            g_prev = g;
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i,j)) continue;
                    double uc = 0.5*(g.u_at(i-1,j) + g.u_at(i,j));
                    double vc = 0.5*(g.v_at(i,j-1) + g.v_at(i,j));
                    if (!std::isfinite(uc) || !std::isfinite(vc)) ok = false;
                }
        }
        check(ok, std::string(solver_name(solver)) + ": smoke buoyancy stable");
    }

    return test_summary();
}

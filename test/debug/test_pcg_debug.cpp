#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include "pressure/pressure.h"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    Config cfg;
    cfg.scenario = "karman";
    cfg.NX       = 64;
    cfg.Lx       = 4.0;
    cfg.Ly       = 1.0;
    cfg.U_inf    = 1.0;
    cfg.Re       = 200;
    cfg.cyl_cx   = 1.0;
    cfg.cyl_cy   = 0.5;
    cfg.cyl_R    = 0.1;
    cfg.t_end    = 0.5;
    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
    cfg.solve_tol   = 1e-6;

    std::cerr << std::scientific << std::setprecision(6);

    // Test 1: Standard CG (identity preconditioner)
    {
        Config c = cfg;
        c.solve_iters = 500;
        c.solver = "cg";
        auto solver = Factory::create(c.solver);
        LFMSimulator sim(c, std::move(solver));

        for (int s = 0; s < 2; s++) sim.step();
        const Grid& g = sim.grid();
        double max_div = 0, max_p = 0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j)) {
                    max_div = std::max(max_div, std::abs(g.divergence(i,j)));
                    max_p = std::max(max_p, std::abs(g.p_at(i,j)));
                }
        std::cerr << "CG: after 2 steps, max_div=" << max_div << " max|p|=" << max_p << "\n";
    }

    // Test 2: CG with 2000 iterations
    {
        Config c = cfg;
        c.solve_iters = 2000;
        c.solver = "cg";
        auto solver = Factory::create(c.solver);
        LFMSimulator sim(c, std::move(solver));
        for (int s = 0; s < 2; s++) sim.step();
        const Grid& g = sim.grid();
        double max_div = 0, max_p = 0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j)) {
                    max_div = std::max(max_div, std::abs(g.divergence(i,j)));
                    max_p = std::max(max_p, std::abs(g.p_at(i,j)));
                }
        std::cerr << "CG(2000): after 2 steps, max_div=" << max_div << " max|p|=" << max_p << "\n";
    }

    // Test 3: Jacobi for comparison
    {
        Config c = cfg;
        c.solve_iters = 2000;
        c.solver = "jacobi";
        auto solver = Factory::create(c.solver);
        LFMSimulator sim(c, std::move(solver));
        for (int s = 0; s < 2; s++) sim.step();
        const Grid& g = sim.grid();
        double max_div = 0, max_p = 0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j)) {
                    max_div = std::max(max_div, std::abs(g.divergence(i,j)));
                    max_p = std::max(max_p, std::abs(g.p_at(i,j)));
                }
        std::cerr << "Jacobi: after 2 steps, max_div=" << max_div << " max|p|=" << max_p << "\n";
    }

    return 0;
}

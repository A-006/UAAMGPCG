// Quick test: does flipping RHS sign fix CG?
#include "config/config.h"
#include "core/grid.h"
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
    cfg.solve_tol = 1e-6;
    cfg.frame_skip = 100;

    std::cerr << std::scientific << std::setprecision(6);

    // Test CG with positive RHS sign
    {
        Config c = cfg;
        c.solve_iters = 200;
        c.solver = "cg";
        auto solver = Factory::create(c.solver);
        ChorinSimulator sim(c, std::move(solver));
        sim.step();
        const Grid& g = sim.grid();
        double max_div = 0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j))
                    max_div = std::max(max_div, std::abs(g.divergence(i,j)));
        std::cerr << "CG: max_div=" << max_div << "\n";
    }

    // Test PCG/GMG
    {
        Config c = cfg;
        c.solve_iters = 50;
        c.solver = "pcg";
        auto solver = Factory::create(c.solver);
        ChorinSimulator sim(c, std::move(solver));
        sim.step();
        const Grid& g = sim.grid();
        double max_div = 0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j))
                    max_div = std::max(max_div, std::abs(g.divergence(i,j)));
        std::cerr << "PCG/GMG: max_div=" << max_div << "\n";
    }

    return 0;
}

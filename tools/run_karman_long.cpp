/**
 * @file run_karman_long.cpp
 * @brief Long Karman vortex street simulation for animation.
 *
 * NX=256, T_END=150s (default), frame_skip=50 for high temporal resolution.
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.scenario = "karman";
    cfg.NX       = 256;
    cfg.Lx       = 4.0;
    cfg.Ly       = 1.0;
    cfg.U_inf    = 1.0;
    cfg.Re       = 200;
    cfg.cyl_cx   = 1.0;
    cfg.cyl_cy   = 0.5;
    cfg.cyl_R    = 0.1;
    cfg.t_end    = 150.0;   // long simulation for rich animation
    cfg.dt       = 0.0;     // auto
    cfg.solve_iters = 100;
    cfg.solve_tol   = 1e-10;
    cfg.frame_skip  = 50;    // more frequent output (every 50 steps)
    cfg.out_dir     = "output_karman";
    cfg.solver      = "pcg_uaamg";

    if (argc > 1) cfg.t_end = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);

    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;

    int nsteps = (int)(cfg.t_end / cfg.dt);
    int nframes = nsteps / cfg.frame_skip + 1;

    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║  Long Karman Vortex Street Simulation             ║\n";
    std::cout << "╠════════════════════════════════════════════════════╣\n";
    std::cout << "║  Grid: " << cfg.NX << "×" << cfg.NY
              << "  |  dx=" << (cfg.Lx / cfg.NX)
              << "  dt=" << cfg.dt << "\n";
    std::cout << "║  t_end=" << cfg.t_end << "  |  steps=" << nsteps
              << "  |  frames=" << nframes << "\n";
    std::cout << "║  frame_skip=" << cfg.frame_skip
              << "  |  output=" << cfg.out_dir << "\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n\n";

    auto solver = Factory::create(cfg.solver);
    ChorinSimulator sim(cfg, std::move(solver));

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < nsteps; s++) {
        sim.step();

        if (s % (nsteps / 20) == 0 && s > 0) {
            const Grid& g = sim.grid();
            double max_div = 0;
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++)
                    if (!g.is_solid(i, j))
                        max_div = std::max(max_div, std::abs(g.divergence(i, j)));

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double eta = elapsed / (s + 1) * nsteps;
            std::cout << "  [" << std::setw(3) << (100 * s / nsteps) << "%]"
                      << "  step=" << s << "/" << nsteps
                      << "  t=" << std::fixed << std::setprecision(2) << ((s + 1) * cfg.dt)
                      << "  max_div=" << std::scientific << std::setprecision(2) << max_div
                      << "  elapsed=" << std::fixed << std::setprecision(0) << elapsed << "s"
                      << "  ETA=" << std::setprecision(0) << eta << "s\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << nsteps << " steps in " << std::fixed
              << std::setprecision(1) << elapsed << " s ("
              << std::setprecision(1) << (elapsed / nsteps * 1000) << " ms/step)\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";

    return 0;
}

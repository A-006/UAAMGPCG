/**
 * @file run_karman_chorin.cpp
 * @brief Chorin reference Karman run (backup if LFM doesn't shed).
 * Usage: run_karman_chorin [t_end] [frame_skip] [NX]
 */
#include "core/config.h"
#include "io/force.h"
#include "simulator/factory.h"
#include "simulator/runner.h"
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    Config cfg;
    cfg.scenario        = "karman";
    cfg.NX              = 256;
    cfg.Lx              = 4.0;
    cfg.Ly              = 1.0;
    cfg.U_inf           = 1.0;
    cfg.Re              = 200;
    cfg.cyl_cx          = 1.0;
    cfg.cyl_cy          = 0.5;
    cfg.cyl_R           = 0.1;
    cfg.t_end           = 40.0;
    cfg.solve_iters     = 100;
    cfg.solve_tol       = 1e-10;
    cfg.frame_skip      = 25;
    cfg.out_dir         = "output_karman_chorin";
    cfg.solver          = "pcg_uaamg";
    cfg.time_integrator = "chorin";

    if (argc > 1)
        cfg.t_end = std::atof(argv[1]);
    if (argc > 2)
        cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3)
        cfg.NX = std::atoi(argv[3]);
    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;

    // Per-frame force diagnostics, plugged into the shared driver loop.
    const double D = 2.0 * cfg.cyl_R;
    sim::RunOptions opts;
    opts.observers.push_back([D](int, const Simulator& s, const Config& c) {
        auto F = computeForce(s.grid(), c.dt, c.U_inf, c.Re, c.cyl_cx, c.cyl_cy, c.cyl_R);
        std::cout << "      Cl=" << std::setw(7) << std::fixed << std::setprecision(3)
                  << F.Cl(c.U_inf, D) << "  Cd=" << std::setw(7) << F.Cd(c.U_inf, D) << "\n";
    });

    auto sim = SimulatorFactory::create(cfg);
    sim::run(*sim, cfg, opts);
    return 0;
}

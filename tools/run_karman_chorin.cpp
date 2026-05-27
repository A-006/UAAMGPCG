/**
 * @file run_karman_chorin.cpp
 * @brief Chorin reference Karman run (backup if LFM doesn't shed).
 * Usage: run_karman_chorin [t_end] [frame_skip] [NX]
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include "io/vtk_writer.h"
#include "force/force.h"
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
    cfg.t_end    = 40.0;
    cfg.solve_iters = 100;
    cfg.solve_tol   = 1e-10;
    cfg.frame_skip  = 25;
    cfg.out_dir     = "output_karman_chorin";
    cfg.solver      = "pcg_uaamg";
    cfg.time_integrator = "chorin";

    if (argc > 1) cfg.t_end     = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) cfg.NX        = std::atoi(argv[3]);
    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;

    double D = 2.0 * cfg.cyl_R;
    int nsteps = (int)(cfg.t_end / cfg.dt);

    std::cout << "Chorin: NX=" << cfg.NX << " NY=" << cfg.NY
              << " dt=" << cfg.dt << " t_end=" << cfg.t_end
              << " nsteps=" << nsteps << "  out=" << cfg.out_dir << "\n";

    mkdir(cfg.out_dir.c_str(), 0755);

    auto solver = Factory::create(cfg.solver);
    ChorinSimulator sim(cfg, std::move(solver));
    VtkWriter::write(sim.grid(), 0, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    int frame = 1;
    for (int s = 1; s <= nsteps; s++) {
        sim.step();
        if (s % cfg.frame_skip == 0) {
            VtkWriter::write(sim.grid(), frame, cfg);
            auto F = computeForce(sim.grid(), cfg.dt, cfg.U_inf, cfg.Re,
                                   cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
            auto t1 = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  frame " << std::setw(4) << frame
                      << "  step " << std::setw(6) << s << "/" << nsteps
                      << "  t=" << std::fixed << std::setprecision(2) << sim.time()
                      << "  Cl=" << std::setw(7) << std::setprecision(3) << F.Cl(cfg.U_inf, D)
                      << "  Cd=" << std::setw(7) << std::setprecision(3) << F.Cd(cfg.U_inf, D)
                      << "  elapsed=" << std::setprecision(0) << el << "s\n";
            frame++;
        }
    }
    return 0;
}

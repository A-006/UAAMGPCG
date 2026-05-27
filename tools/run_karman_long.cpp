/**
 * @file run_karman_long.cpp
 * @brief Long Karman vortex street simulation using LFM (paper's Algorithm 1).
 *
 * Usage: run_karman_long [t_end] [frame_skip] [NX]
 *   defaults: t_end=40, frame_skip=10, NX=256
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/factory.h"
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
    cfg.solve_iters = 200;
    cfg.solve_tol   = 1e-10;
    cfg.frame_skip  = 10;
    cfg.out_dir     = "output_karman";
    cfg.solver      = "pcg_uaamg";
    cfg.time_integrator = "lfm";
    cfg.lfm_cycle_steps = 2;

    if (argc > 1) cfg.t_end     = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) cfg.NX        = std::atoi(argv[3]);

    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.25 * (cfg.Lx / cfg.NX) / cfg.U_inf;

    double dx = cfg.Lx / cfg.NX;
    double D  = 2.0 * cfg.cyl_R;
    int nsteps = (int)(cfg.t_end / (cfg.dt * cfg.lfm_cycle_steps));
    int nframes = nsteps / cfg.frame_skip + 1;

    std::cout << "===================================================\n";
    std::cout << "  LFM Karman Vortex Street (Algorithm 1)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "x" << cfg.NY
              << "   dx=" << dx << "   D/dx=" << (D/dx) << "\n";
    std::cout << "  dt=" << cfg.dt << "   cycle_steps=" << cfg.lfm_cycle_steps
              << "   cycle_dt=" << (cfg.dt*cfg.lfm_cycle_steps) << "\n";
    std::cout << "  t_end=" << cfg.t_end << "   cycles=" << nsteps
              << "   frames~" << nframes << "\n";
    std::cout << "  Re=" << cfg.Re << "   output=" << cfg.out_dir << "/\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);

    auto pressure_solver = SimulatorFactory::make_pressure_solver(cfg);
    auto sim_ptr = SimulatorFactory::create(cfg, std::move(pressure_solver));
    Simulator& sim = *sim_ptr;

    // Initial frame
    VtkWriter::write(sim.grid(), 0, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    int frame = 1;
    for (int s = 1; s <= nsteps; s++) {
        sim.step();

        if (s % cfg.frame_skip == 0) {
            VtkWriter::write(sim.grid(), frame, cfg);
            const Grid& g = sim.grid();
            double max_div = 0, max_u = 0;
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i, j)) continue;
                    max_div = std::max(max_div, std::abs(g.divergence(i, j)));
                    double uc = 0.5*(g.u_at(i,j)+g.u_at(i-1,j));
                    double vc = 0.5*(g.v_at(i,j)+g.v_at(i,j-1));
                    max_u = std::max(max_u, std::sqrt(uc*uc+vc*vc));
                }
            auto force = computeForce(g, cfg.dt, cfg.U_inf, cfg.Re,
                                       cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  frame " << std::setw(4) << frame
                      << "  cycle " << std::setw(5) << s << "/" << nsteps
                      << "  t=" << std::fixed << std::setprecision(2) << sim.time()
                      << "  Cl=" << std::setw(7) << std::setprecision(3) << force.Cl(cfg.U_inf, D)
                      << "  Cd=" << std::setw(7) << std::setprecision(3) << force.Cd(cfg.U_inf, D)
                      << "  |u|max=" << std::setprecision(2) << max_u
                      << "  div=" << std::scientific << std::setprecision(1) << max_div
                      << "  elapsed=" << std::fixed << std::setprecision(0) << elapsed << "s\n";
            frame++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << nsteps << " cycles, " << (frame-1) << " frames in "
              << std::fixed << std::setprecision(1) << elapsed << " s\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";

    return 0;
}

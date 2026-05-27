/**
 * @file run_vortex_ring_collision.cpp
 * @brief Head-on vortex ring collision (paper Fig. 3) — 3D demonstration.
 *
 * Two coaxial vortex rings approach each other along the x-axis. They
 * expand outward as they collide. This exercises the full 3D pipeline:
 * Biot-Savart initial condition → semi-Lagrangian advection →
 * 3D Poisson projection (UAAMG) → free-slip walls.
 *
 * Usage: run_vortex_ring_collision [t_end] [frame_skip] [NX]
 *   defaults: t_end=2.0, frame_skip=2, NX=64
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "simulator/simulator_3d.h"
#include "solver/factory_3d.h"
#include "io/vtk_writer_3d.h"
#include "scenarios/vortex_ring.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim       = 3;
    cfg.NX = cfg.NY = cfg.NZ = 64;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf  = 1.0;
    cfg.Re     = 1000.0;
    cfg.cyl_R  = 0.1;             // reused as a length scale for viscosity
    cfg.dt     = 0.005;
    cfg.t_end  = 2.0;
    cfg.solve_iters = 200;
    cfg.solve_tol   = 1e-8;
    cfg.frame_skip  = 2;
    cfg.out_dir     = "output_vortex_ring";
    cfg.solver      = "cg";        // 3D pcg_uaamg isn't wired yet; cg works
    cfg.time_integrator = "chorin";

    if (argc > 1) cfg.t_end     = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[3]);

    int nsteps  = (int)(cfg.t_end / cfg.dt);
    int nframes = nsteps / cfg.frame_skip + 1;

    std::cout << "===================================================\n";
    std::cout << "  3D Vortex Ring Head-on Collision (paper Fig. 3)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3  dx=" << (cfg.Lx / cfg.NX) << "\n";
    std::cout << "  dt=" << cfg.dt << "  t_end=" << cfg.t_end
              << "  steps=" << nsteps << "  frames~" << nframes << "\n";
    std::cout << "  Solver: " << cfg.solver << "  out=" << cfg.out_dir << "/\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory3D::create(cfg.solver);
    ChorinSimulator3D sim(cfg, std::move(solver));

    // ── Two opposing vortex rings facing each other along x ──
    scenarios::VortexRing left  = { {0.30, 0.50, 0.50}, {1, 0, 0}, 0.15, 0.03, +0.5, 200 };
    scenarios::VortexRing right = { {0.70, 0.50, 0.50}, {1, 0, 0}, 0.15, 0.03, -0.5, 200 };
    scenarios::add_vortex_ring(sim.mutable_grid(), left);
    scenarios::add_vortex_ring(sim.mutable_grid(), right);

    VtkWriter3D::write(sim.grid(), 0, cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    int frame = 1;
    for (int s = 1; s <= nsteps; s++) {
        sim.step();
        if (s % cfg.frame_skip == 0) {
            VtkWriter3D::write(sim.grid(), frame, cfg);
            VtkWriter3D::printStatus(s, sim.time(), sim.grid());
            frame++;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << nsteps << " steps, " << (frame - 1)
              << " frames in " << std::fixed << std::setprecision(1) << el << " s\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";
    return 0;
}

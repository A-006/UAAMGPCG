/**
 * @file run_vortex_ring_connection.cpp
 * @brief Vortex ring connection / reconnection (paper Fig. 5).
 *
 * Two coplanar but offset vortex rings with same-sign circulation are
 * placed at a slight tilt so their nearest sides approach. As they
 * propagate they deform, their near-cores annihilate, and a new merged
 * vortex topology forms (the classical reconnection scenario).
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
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 64;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf = 1.0;
    cfg.Re = 2000.0;
    cfg.cyl_R = 0.1;
    cfg.dt = 0.004;
    cfg.t_end = 2.0;
    cfg.solve_iters = 200;
    cfg.solve_tol = 1e-8;
    cfg.frame_skip = 4;
    cfg.out_dir = "output_vortex_connection";
    cfg.solver = "cg";

    if (argc > 1) cfg.t_end = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[3]);

    int nsteps = (int)(cfg.t_end / cfg.dt);
    std::cout << "===================================================\n";
    std::cout << "  3D Vortex Ring Connection (paper Fig. 5)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3   dt=" << cfg.dt
              << "   t_end=" << cfg.t_end << "   steps=" << nsteps << "\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory3D::create(cfg.solver);
    ChorinSimulator3D sim(cfg, std::move(solver));

    // Two same-sign rings offset perpendicular to their axes, tilted
    // 25° so their near sides approach and reconnect.
    double s25 = std::sin(25.0 * M_PI / 180.0);
    double c25 = std::cos(25.0 * M_PI / 180.0);
    scenarios::VortexRing ringA = {
        {0.40, 0.50, 0.50}, {c25, +s25, 0}, 0.14, 0.025, +0.6, 240
    };
    scenarios::VortexRing ringB = {
        {0.60, 0.50, 0.50}, {c25, -s25, 0}, 0.14, 0.025, +0.6, 240
    };
    scenarios::add_vortex_ring(sim.mutable_grid(), ringA);
    scenarios::add_vortex_ring(sim.mutable_grid(), ringB);

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
    std::cout << "\n  Done: " << (frame - 1) << " frames in "
              << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    return 0;
}

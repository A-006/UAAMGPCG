/**
 * @file run_trefoil_knot.cpp
 * @brief Trefoil knot vortex evolution (paper Fig. 7).
 *
 * A trefoil-knot-shaped vortex filament is set up via Biot-Savart and
 * evolved by ChorinSimulator3D. The paper shows the knot breaks into a
 * large vortex and a small vortex as it relaxes.
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "simulator/simulator_3d.h"
#include "solver/factory_3d.h"
#include "io/vtk_writer_3d.h"
#include "scenarios/trefoil_knot.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 64;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf = 1.0;
    cfg.Re = 2000.0;
    cfg.cyl_R = 0.05;
    cfg.dt = 0.004;
    cfg.t_end = 4.0;
    cfg.solve_iters = 200;
    cfg.solve_tol = 1e-8;
    cfg.frame_skip = 5;
    cfg.out_dir = "output_trefoil";
    cfg.solver = "cg";

    if (argc > 1) cfg.t_end = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[3]);

    int nsteps = (int)(cfg.t_end / cfg.dt);
    std::cout << "===================================================\n";
    std::cout << "  3D Trefoil Knot Vortex (paper Fig. 7)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3   dt=" << cfg.dt
              << "   t_end=" << cfg.t_end << "   steps=" << nsteps << "\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory3D::create(cfg.solver);
    ChorinSimulator3D sim(cfg, std::move(solver));

    scenarios::TrefoilKnot tk{};
    tk.center = { 0.5, 0.5, 0.5 };
    tk.scale  = 0.07;
    tk.core   = 0.025;
    tk.circulation = 0.5;
    tk.n_segments = 360;
    scenarios::add_trefoil_knot(sim.mutable_grid(), tk);

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

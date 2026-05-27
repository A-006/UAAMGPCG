/**
 * @file run_delta_wing.cpp
 * @brief Delta wing in uniform inflow (paper Fig. 1 left).
 *
 * A triangular wing at angle-of-attack sheds wingtip vortices into a
 * uniform stream. Demonstrates immersed solids + inflow/outflow BCs in 3D.
 */
#include "config/config.h"
#include "simulator/simulator_3d.h"
#include "solver/factory_3d.h"
#include "io/vtk_writer_3d.h"
#include "scenarios/delta_wing.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX = 96; cfg.NY = 48; cfg.NZ = 64;
    cfg.Lx = 3.0; cfg.Ly = 1.5; cfg.Lz = 2.0;
    cfg.U_inf = 1.0;
    cfg.Re    = 5000.0;
    cfg.cyl_R = 0.2;          // length scale for viscosity (≈ root chord)
    cfg.dt    = 0.005;
    cfg.t_end = 5.0;
    cfg.solve_iters = 200;
    cfg.solve_tol = 1e-7;
    cfg.frame_skip = 5;
    cfg.out_dir = "output_delta_wing";
    cfg.solver = "cg";

    if (argc > 1) cfg.t_end = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) { int n = std::atoi(argv[3]); cfg.NX = n; cfg.NY = n/2; cfg.NZ = 2*n/3; }

    int nsteps = (int)(cfg.t_end / cfg.dt);
    std::cout << "===================================================\n";
    std::cout << "  3D Delta Wing Vortices (paper Fig. 1)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ
              << "   dt=" << cfg.dt << "   t_end=" << cfg.t_end << "\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory3D::create(cfg.solver);
    ChorinSimulator3D sim(cfg, std::move(solver));

    // Wing + inflow/outflow BCs + initial uniform stream.
    scenarios::DeltaWing wing{};
    wing.leading_x = 0.7;
    wing.chord     = 0.8;
    wing.semi_span = 0.4;
    wing.thickness = 0.025;
    wing.aoa_deg   = 15.0;
    wing.y_mid     = 0.75;
    scenarios::setup_delta_wing(sim.mutable_grid(), wing);

    sim.set_boundary_manager(scenarios::delta_wing_bcs(cfg.U_inf));
    scenarios::set_uniform_inflow(sim.mutable_grid(), cfg.U_inf);

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

/**
 * @file run_wind_turbine.cpp
 * @brief Three-blade wind turbine in inflow (paper Fig. 2).
 *
 * At each step the rotor angle is advanced by ω·Δt and the moving solid
 * mask + rigid-body rotational velocity on the blade surface are
 * re-applied. The downstream helical trail emerges from the moving
 * wing-tip vortices.
 */
#include "config/config.h"
#include "simulator/simulator_3d.h"
#include "solver/factory_3d.h"
#include "io/vtk_writer_3d.h"
#include "scenarios/wind_turbine.h"
#include "scenarios/delta_wing.h"   // set_uniform_inflow
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX = 96; cfg.NY = 64; cfg.NZ = 64;
    cfg.Lx = 3.0; cfg.Ly = 2.0; cfg.Lz = 2.0;
    cfg.U_inf = 1.0;
    cfg.Re = 3000.0;
    cfg.cyl_R = 0.5;
    cfg.dt = 0.004;
    cfg.t_end = 4.0;
    cfg.solve_iters = 200;
    cfg.solve_tol = 1e-7;
    cfg.frame_skip = 5;
    cfg.out_dir = "output_wind_turbine";
    cfg.solver = "cg";

    if (argc > 1) cfg.t_end = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);

    int nsteps = (int)(cfg.t_end / cfg.dt);
    std::cout << "===================================================\n";
    std::cout << "  3D Wind Turbine (paper Fig. 2)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ
              << "   dt=" << cfg.dt << "   t_end=" << cfg.t_end << "\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory3D::create(cfg.solver);
    ChorinSimulator3D sim(cfg, std::move(solver));

    scenarios::WindTurbine wt{};
    wt.hub_center      = { 1.0, 1.0, 1.0 };
    wt.blade_radius    = 0.45;
    wt.hub_radius      = 0.05;
    wt.chord           = 0.08;
    wt.thickness       = 0.025;
    wt.n_blades        = 3;
    wt.angular_velocity = 4.0;     // ~0.64 Hz

    sim.set_boundary_manager(scenarios::wind_turbine_bcs(cfg.U_inf));
    scenarios::set_uniform_inflow(sim.mutable_grid(), cfg.U_inf);
    scenarios::apply_turbine_state(sim.mutable_grid(), wt, 0.0);

    VtkWriter3D::write(sim.grid(), 0, cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    int frame = 1;
    double angle = 0.0;
    for (int s = 1; s <= nsteps; s++) {
        angle += wt.angular_velocity * cfg.dt;
        scenarios::apply_turbine_state(sim.mutable_grid(), wt, angle);
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

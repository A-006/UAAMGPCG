/**
 * @file run_fire_ball.cpp
 * @brief Buoyancy-driven fire ball / thermal plume (paper Fig. 4).
 *
 * A Gaussian hot region is seeded near the bottom and buoyancy drives
 * it upward. The temperature field is advected by the flow at every
 * step (semi-Lagrangian), and the flow is forced by Boussinesq buoyancy
 * proportional to (T - T_ref). Vorticity organizes naturally as the
 * plume rises.
 */
#include "config/config.h"
#include "simulator/simulator_3d.h"
#include "solver/factory_3d.h"
#include "io/vtk_writer_3d.h"
#include "scalar/scalar_field_3d.h"
#include "scenarios/fire_ball.h"
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
    cfg.cyl_R = 0.1;
    cfg.dt = 0.004;
    cfg.t_end = 3.0;
    cfg.solve_iters = 200;
    cfg.solve_tol = 1e-7;
    cfg.frame_skip = 5;
    cfg.out_dir = "output_fire_ball";
    cfg.solver = "cg";

    double beta = 5.0;          // buoyancy coefficient
    double T_ref = 0.0;

    if (argc > 1) cfg.t_end = std::atof(argv[1]);
    if (argc > 2) cfg.frame_skip = std::atoi(argv[2]);
    if (argc > 3) cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[3]);

    int nsteps = (int)(cfg.t_end / cfg.dt);
    std::cout << "===================================================\n";
    std::cout << "  3D Fire Ball / Thermal Plume (paper Fig. 4)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3   dt=" << cfg.dt
              << "   buoyancy beta=" << beta << "\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory3D::create(cfg.solver);
    ChorinSimulator3D sim(cfg, std::move(solver));

    // Default free-slip box BCs + buoyancy along +z (apply_buoyancy pushes w).
    // Hot region starts near the floor (z low).
    ScalarField3D T(sim.grid());
    ScalarField3D T_next(sim.grid());
    scenarios::FireBall fb{};
    fb.center = { 0.5, 0.5, 0.20 };
    fb.radius = 0.08;
    fb.T_hot  = 1.0;
    fb.T_ref  = T_ref;
    scenarios::seed_fire_ball(T, sim.grid(), fb);

    VtkWriter3D::write(sim.grid(), 0, cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    int frame = 1;
    for (int s = 1; s <= nsteps; s++) {
        // Advect temperature with the current velocity field
        ScalarField3D::advect(T, T_next, sim.grid(), cfg.dt);
        std::swap(T.data(), T_next.data());

        // Apply buoyancy to v-velocity (use w-direction for +z gravity? Here +y).
        apply_buoyancy(sim.mutable_grid(), T, T_ref, beta, cfg.dt);

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

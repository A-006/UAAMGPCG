/**
 * @file run_delta_wing_lfm.cpp
 * @brief Delta wing via the LFM method (paper Fig. 9) — PHASE 1 pipeline check.
 *
 * Validates that our LFM cycle + immersed solid + inflow/outflow BC + BFECC
 * clamp (inviscid) runs stably and sheds wingtip vortices. Uses the paper's
 * domain aspect (2x1x1) and method recipe (inviscid + BFECC clamp + n=5), with
 * our analytic delta-wing geometry. Phase 2 swaps in the authors' exact
 * solid_sdf.npy + 20-degree freestream at full 256x128x128.
 *
 * Usage: run_delta_wing_lfm [cycles] [NX] [dt] [out_dir]
 *   defaults: cycles=160 NX=96 dt=0.005 out=output_delta_wing_lfm
 */
#include "core/config.h"
#include "io/vtk_writer_3d.h"
#include "integrator/lfm/lfm_simulator_3d.h"
#include "io/3d/plate.h"
#include "io/3d/freestream.h"
#include "solver/factory_3d.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX  = 96; // domain aspect 2:1:1 (paper)
    cfg.NY  = 48;
    cfg.NZ  = 48;
    cfg.Lx  = 2.0;
    cfg.Ly  = 1.0;
    cfg.Lz  = 1.0;
    cfg.U_inf           = 0.6; // paper freestream |U| = 0.6
    cfg.Re              = 0;   // INVISCID — stability from BFECC clamp
    cfg.lfm_bfecc_clamp = true;
    cfg.dt              = 0.005;
    cfg.solve_iters     = 100;
    cfg.solve_tol       = 1e-7;
    cfg.solver          = "cg";
    cfg.time_integrator = "lfm";
    cfg.lfm_cycle_steps = 5; // paper n = 5
    cfg.out_dir         = "output_delta_wing_lfm";

    int n_cycles = 160;
    if (argc > 1)
        n_cycles = std::atoi(argv[1]);
    if (argc > 2) {
        cfg.NX = std::atoi(argv[2]);
        cfg.NY = cfg.NX / 2;
        cfg.NZ = cfg.NX / 2;
    }
    if (argc > 3)
        cfg.dt = std::atof(argv[3]);
    if (argc > 4)
        cfg.out_dir = argv[4];

    double dx = cfg.Lx / cfg.NX;
    std::cout << "=== LFM Delta Wing (Phase 1 pipeline check) ===\n";
    std::cout << "  Grid " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ << "  dx=" << dx
              << "  INVISCID+clamp  n=" << cfg.lfm_cycle_steps << "  U=" << cfg.U_inf
              << "  dt=" << cfg.dt << " (CFL~" << cfg.U_inf * cfg.dt / dx << ")\n";
    std::cout << "  cycles=" << n_cycles << "  out=" << cfg.out_dir << "/\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    LFMSimulator3D sim(cfg, Factory3D::create(cfg.solver));

    // Analytic delta wing scaled into the 2x1x1 box, at 20-degree AoA.
    scenarios::Plate wing;
    wing.leading_x = 0.5;
    wing.chord     = 1.0;
    wing.semi_span = 0.35;
    wing.thickness = 0.02;
    wing.tilt_deg   = 20.0;
    wing.y_mid     = 0.5;
    // Paper-style freestream at 20-degree attack angle (mass-balanced box BC).
    double aoa = 20.0 * M_PI / 180.0;
    double Ux  = cfg.U_inf * std::cos(aoa); // 0.5638 at U=0.6
    double Uy  = cfg.U_inf * std::sin(aoa); // 0.2052 at U=0.6
    scenarios::setup_plate(sim.mutable_grid(), wing);
    scenarios::set_uniform_freestream(sim.mutable_grid(), Ux, Uy, 0.0);
    sim.set_boundary_manager(scenarios::freestream_box_bcs(Ux, Uy, 0.0));

    VtkWriter3D::write(sim.grid(), 0, cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    int frame = 1;
    for (int c = 1; c <= n_cycles; c++) {
        sim.step();
        if (c % 4 == 0 || c == n_cycles) {
            VtkWriter3D::write(sim.grid(), frame++, cfg);
            VtkWriter3D::printStatus(c, sim.time(), sim.grid());
        }
    }
    auto t1   = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << n_cycles << " cycles in " << std::fixed << std::setprecision(1)
              << el << " s  (" << el / n_cycles << " s/cycle)\n";
    return 0;
}

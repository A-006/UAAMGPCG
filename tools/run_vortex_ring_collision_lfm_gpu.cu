/**
 * @file run_vortex_ring_collision_lfm_gpu.cu
 * @brief Head-on vortex-ring collision (paper Fig. 3), 3D LFM + GPU Poisson.
 *
 * Two coaxial rings of opposite circulation approach along the x-axis, collide
 * at the mid-plane, and expand radially in the y-z plane — the canonical
 * low-dissipation showcase for the LFM integrator (Sun et al. SIGGRAPH 2025).
 * The CPU runs the flow-map marching / pullback; every pressure projection is
 * offloaded to the GPU via `CudaPCGSolver3D` (UAAMG-preconditioned CG). Frames
 * are written as structured-points VTK (velocity + vorticity magnitude).
 *
 * Usage: run_vortex_ring_collision_lfm_gpu [n_cycles] [NX] [circulation] [Re]
 *   defaults: n_cycles=120, NX=96, circulation=1.0, Re=0 (inviscid)
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "io/vtk_writer_3d.h"
#include "numerics/bc/patches_3d.h"
#include "simulator/lfm_simulator_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include "solver/cuda_pcg_solver_3d.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 96;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf                = 1.0;
    cfg.Re                   = 0; // inviscid: crisp, low-dissipation collision
    cfg.cyl_R                = 0.1;
    cfg.dt                   = 0.001;
    cfg.solve_iters          = 200;
    cfg.solve_tol            = 1e-6;
    cfg.time_integrator      = "lfm";
    cfg.lfm_cycle_steps      = 2;
    cfg.out_dir              = "output_vortex_collision_lfm";

    int n_cycles = 120;
    double circ  = 1.0;
    if (argc > 1)
        n_cycles = std::atoi(argv[1]);
    if (argc > 2)
        cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[2]);
    if (argc > 3)
        circ = std::atof(argv[3]);
    if (argc > 4)
        cfg.Re = std::atof(argv[4]);

    double dx       = cfg.Lx / cfg.NX;
    double cycle_dt = cfg.lfm_cycle_steps * cfg.dt;

    std::cout << "===================================================\n";
    std::cout << "  3D LFM Vortex-Ring Head-on Collision (GPU Poisson)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3  dx=" << dx << "\n";
    std::cout << "  dt=" << cfg.dt << "  cycle=" << cfg.lfm_cycle_steps << " steps (" << cycle_dt
              << "/cycle)  cycles=" << n_cycles << "\n";
    std::cout << "  Re=" << cfg.Re << (cfg.Re > 0 ? "" : " (inviscid)")
              << "  circulation=±" << circ << "\n";

    auto solver = std::make_unique<CudaPCGSolver3D>(/*use_precond=*/true);
    std::cout << "  Solver: " << solver->name() << "  out=" << cfg.out_dir << "/\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    LFMSimulator3D sim(cfg, std::move(solver));

    // Two coaxial rings on the x-axis, opposite circulation → head-on approach.
    scenarios::VortexRing left;
    left.center      = {0.32 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
    left.axis        = {1.0, 0.0, 0.0};
    left.radius      = 0.15 * cfg.Lx;
    left.core        = 0.03 * cfg.Lx;
    left.circulation = +circ;
    left.n_segments  = 240;

    scenarios::VortexRing right = left;
    right.center      = {0.68 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
    right.circulation = -circ;

    scenarios::add_vortex_ring(sim.mutable_grid(), left);
    scenarios::add_vortex_ring(sim.mutable_grid(), right);
    sim.set_boundary_manager(bc::free_slip_box()); // re-apply BC after injecting IC

    VtkWriter3D::write(sim.grid(), 0, cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int c = 1; c <= n_cycles; c++) {
        sim.step();
        VtkWriter3D::write(sim.grid(), c, cfg);
        if (c % 5 == 0 || c == n_cycles)
            VtkWriter3D::printStatus(c, sim.time(), sim.grid());
    }
    auto t1   = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << n_cycles << " cycles, " << (n_cycles + 1) << " frames in "
              << std::fixed << std::setprecision(1) << el << " s\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";
    return 0;
}

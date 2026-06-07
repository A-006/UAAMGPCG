/**
 * @file run_vortex_ring_lfm_gpu.cu
 * @brief 3D LFM vortex-ring evolution, GPU-accelerated Poisson solve.
 *
 * Drives `LFMSimulator3D` (the impulse/flow-map integrator, Sun et al. 2025
 * Algorithm 1 in 3D) on a single Gaussian-cored vortex ring inside a free-slip
 * box. The CPU runs the flow-map marching / pullback; every pressure
 * projection is offloaded to the GPU via `CudaPCGSolver3D` (UAAMG-preconditioned
 * CG). Frames are written as structured-points VTK (velocity + vorticity
 * magnitude + divergence) for visualization.
 *
 * Usage: run_vortex_ring_lfm_gpu [n_cycles] [NX] [circulation]
 *   defaults: n_cycles=60, NX=64, circulation=1.0
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
    cfg.NX = cfg.NY = cfg.NZ = 64;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf                = 1.0;
    cfg.Re                   = 0; // inviscid: crisp, low-dissipation ring (LFM's strength)
    cfg.cyl_R                = 0.1;
    cfg.dt                   = 0.002;
    cfg.solve_iters          = 200;
    cfg.solve_tol            = 1e-6;
    cfg.time_integrator      = "lfm";
    cfg.lfm_cycle_steps      = 2;
    cfg.out_dir              = "output_vortex_ring_lfm";

    int n_cycles   = 60;
    double circ    = 1.0;
    if (argc > 1)
        n_cycles = std::atoi(argv[1]);
    if (argc > 2)
        cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[2]);
    if (argc > 3)
        circ = std::atof(argv[3]);

    double dx          = cfg.Lx / cfg.NX;
    double cycle_dt    = cfg.lfm_cycle_steps * cfg.dt;

    std::cout << "===================================================\n";
    std::cout << "  3D LFM Vortex Ring (GPU Poisson)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3  dx=" << dx << "\n";
    std::cout << "  dt=" << cfg.dt << "  cycle=" << cfg.lfm_cycle_steps << " steps (" << cycle_dt
              << "/cycle)  cycles=" << n_cycles << "\n";
    std::cout << "  Re=" << cfg.Re << " (inviscid)  circulation=" << circ << "\n";

    auto solver = std::make_unique<CudaPCGSolver3D>(/*use_precond=*/true);
    std::cout << "  Solver: " << solver->name() << "  out=" << cfg.out_dir << "/\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    LFMSimulator3D sim(cfg, std::move(solver));

    // Single ring low in the box, propagating along +z toward the far wall.
    scenarios::VortexRing ring;
    ring.center      = {0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.28 * cfg.Lz};
    ring.axis        = {0.0, 0.0, 1.0};
    ring.radius      = 0.18 * cfg.Lx;
    ring.core        = 0.045 * cfg.Lx;
    ring.circulation = circ;
    ring.n_segments  = 240;
    scenarios::add_vortex_ring(sim.mutable_grid(), ring);
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

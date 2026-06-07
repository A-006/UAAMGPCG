/**
 * @file run_vortex_ring_collision_lfm_gpu_resident.cu
 * @brief Head-on vortex-ring collision (paper Fig. 3), FULLY GPU-resident LFM.
 *
 * Two coaxial rings of opposite circulation approach along x, collide, and
 * expand radially. Unlike run_vortex_ring_collision_lfm_gpu.cu (CPU cycle + GPU
 * Poisson), the ENTIRE LFM cycle runs on the GPU via CudaLFMSimulator3D, copying
 * back to host only once per cycle for VTK output.
 *
 * Usage: run_vortex_ring_collision_lfm_gpu_resident [n_cycles] [NX] [circ] [Re]
 *   defaults: n_cycles=120, NX=96, circulation=1.0, Re=0 (inviscid)
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "io/vtk_writer_3d.h"
#include "numerics/ops/operators_3d.h"
#include "simulator/cuda_lfm_simulator_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

// Slim VTK: vorticity-magnitude scalar only (structured points). At high
// resolution the full VtkWriter3D (velocity vectors + extra scalars) is ~0.5–4.5
// GB/frame, which fills the disk; the iso-surface renderer only needs |ω|.
static void write_vort_vtk(const Grid3D& g, int frame, const std::string& dir) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/frame_%05d.vtk", dir.c_str(), frame);
    std::ofstream f(path);
    f << "# vtk DataFile Version 2.0\nLFM 3D vorticity - Frame " << frame
      << "\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << g.nx + 1 << " " << g.ny + 1 << " " << g.nz + 1 << "\n";
    f << "ORIGIN 0 0 0\nSPACING " << g.dx << " " << g.dy << " " << g.dz << "\n";
    long npts = (long)(g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    f << "POINT_DATA " << npts << "\nSCALARS vorticity_magnitude float 1\nLOOKUP_TABLE default\n";
    // Node values = vorticity magnitude at the nearest interior cell (clamped).
    for (int k = 0; k <= g.nz; k++)
        for (int j = 0; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++) {
                int ci = i < 1 ? 1 : (i > g.nx ? g.nx : i);
                int cj = j < 1 ? 1 : (j > g.ny ? g.ny : j);
                int ck = k < 1 ? 1 : (k > g.nz ? g.nz : k);
                f << (float)fvc::vorticity_magnitude(g, ci, cj, ck) << "\n";
            }
}

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 96;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf                = 1.0;
    cfg.Re                   = 0; // inviscid
    cfg.cyl_R                = 0.1;
    cfg.dt                   = 0.001;
    cfg.solve_iters          = 120;
    cfg.solve_tol            = 1e-6;
    cfg.time_integrator      = "lfm";
    cfg.lfm_cycle_steps      = 2;
    cfg.out_dir              = "output_vortex_collision_lfm_gpu";

    int n_cycles     = 120;
    double circ      = 1.0;
    // Paper-faithful default: NO artificial seed — the burst into secondary
    // vortex filaments develops naturally from the grid's own anisotropy at
    // sufficient resolution (the paper describes no perturbation). The seed
    // args remain for experimentation.
    int perturb_n    = 0;
    double perturb_a = 0.0;
    if (argc > 1)
        n_cycles = std::atoi(argv[1]);
    if (argc > 2)
        cfg.NX = cfg.NY = cfg.NZ = std::atoi(argv[2]);
    if (argc > 3)
        circ = std::atof(argv[3]);
    if (argc > 4)
        cfg.Re = std::atof(argv[4]);
    if (argc > 5)
        perturb_n = std::atoi(argv[5]);
    if (argc > 6)
        perturb_a = std::atof(argv[6]);
    if (argc > 7)
        cfg.dt = std::atof(argv[7]); // override dt (CFL control for high-res/violent collision)
    if (argc > 8)
        cfg.out_dir = argv[8]; // output directory (so parallel runs don't collide)
    int frame_skip = 4;        // write a (slim) frame every N cycles — keep disk bounded
    if (argc > 9)
        frame_skip = std::atoi(argv[9]);

    std::cout << "===================================================\n";
    std::cout << "  3D LFM Vortex-Ring Collision — FULLY GPU-RESIDENT\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid: " << cfg.NX << "^3  dx=" << (cfg.Lx / cfg.NX) << "  cycles=" << n_cycles
              << "  Re=" << cfg.Re << "  circ=±" << circ << "  seed: n=" << perturb_n << " amp="
              << perturb_a << "\n";
    std::cout << "  out=" << cfg.out_dir << "/\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    CudaLFMSimulator3D sim(cfg);

    // Two coaxial rings on the x-axis, opposite circulation → head-on collision.
    // Rings are kept small so the post-collision ring (which expands to ~3–4×
    // its initial radius) bursts in open space without hitting the free-slip
    // walls. Smooth filament (n_segments=400).
    scenarios::VortexRing left;
    left.center      = {0.36 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
    left.axis        = {1.0, 0.0, 0.0};
    left.radius      = 0.10 * cfg.Lx;
    left.core        = 0.025 * cfg.Lx;
    left.circulation = +circ;
    left.n_segments  = 200;
    left.perturb_n   = perturb_n;
    left.perturb_amp = perturb_a;
    scenarios::VortexRing right = left;
    right.center                = {0.64 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
    right.circulation           = -circ;
    scenarios::add_vortex_ring(sim.mutable_grid(), left);
    scenarios::add_vortex_ring(sim.mutable_grid(), right);
    sim.commit();

    int frame = 0;
    write_vort_vtk(sim.grid(), frame++, cfg.out_dir); // slim |ω|-only VTK (disk-friendly)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int c = 1; c <= n_cycles; c++) {
        sim.step();
        if (c % frame_skip == 0 || c == n_cycles)
            write_vort_vtk(sim.grid(), frame++, cfg.out_dir);
        if (c % 5 == 0 || c == n_cycles)
            VtkWriter3D::printStatus(c, sim.time(), sim.grid());
    }
    auto t1   = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << n_cycles << " cycles in " << std::fixed << std::setprecision(1)
              << el << " s  (" << el / n_cycles << " s/cycle)\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";
    return 0;
}

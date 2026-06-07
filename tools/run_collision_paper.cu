/**
 * @file run_collision_paper.cu
 * @brief Paper-faithful head-on vortex-ring collision (LFM paper Fig. 3).
 *
 * Reproduces the authors' actual recipe (from their source + paper Table 4):
 *   - grid 128 x 256 x 256  (collision axis = the SHORT 128 dir; rings expand
 *     into the large 256x256 plane — paper's aspect, NOT a cube)
 *   - INVISCID (no physical viscosity)
 *   - BFECC neighbour clamp ON  → the inviscid cycle stays stable (paper's
 *     BfeccClamp; this is the "numerical viscosity" that lets it burst w/o ν)
 *   - n = 5 steps per reinitialization cycle, CG fixed at 8 iterations
 * Domain 0.5 x 1 x 1 keeps dx=dy=dz uniform (= 1/256).
 *
 * Usage: run_collision_paper [cycles] [dt] [out_dir] [frame_skip]
 *   defaults: cycles=250 dt=4e-4 out=output_collision_paper frame_skip=4
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "io/vtk_writer_3d.h"
#include "numerics/ops/operators_3d.h"
#include "simulator/cuda_lfm_simulator_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

// Slim |ω|-only structured-points VTK (disk-friendly; iso/volume render needs only |ω|).
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
    // Paper aspect: collision axis x is the SHORT 128; rings expand into 256x256.
    cfg.NX = 128;
    cfg.NY = 256;
    cfg.NZ = 256;
    cfg.Lx = 0.5; // 128/256 → dx = dy = dz = 1/256 (uniform)
    cfg.Ly = 1.0;
    cfg.Lz = 1.0;
    cfg.U_inf           = 1.0;
    cfg.Re              = 0;    // INVISCID — stability comes from the BFECC clamp
    cfg.lfm_bfecc_clamp = true; // paper's BfeccClamp (numerical viscosity)
    cfg.cyl_R           = 0.1;
    cfg.dt              = 4e-4;
    cfg.solve_iters     = 8; // paper: CG fixed at 8 iterations
    cfg.solve_tol       = 0.0;
    cfg.time_integrator = "lfm";
    cfg.lfm_cycle_steps = 5; // paper: n = 5 steps per reinitialization cycle
    cfg.out_dir         = "output_collision_paper";

    int n_cycles = 250;
    if (argc > 1)
        n_cycles = std::atoi(argv[1]);
    if (argc > 2)
        cfg.dt = std::atof(argv[2]);
    if (argc > 3)
        cfg.out_dir = argv[3];
    int frame_skip = 4;
    if (argc > 4)
        frame_skip = std::atoi(argv[4]);
    if (argc > 5)
        cfg.lfm_bfecc_clamp = std::atoi(argv[5]) != 0; // diagnostic: toggle clamp

    double dx = cfg.Lx / cfg.NX;
    std::cout << "===================================================\n";
    std::cout << "  Paper-faithful Head-on Vortex Collision (Fig. 3)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ << "  dx=" << dx
              << " (uniform)\n";
    std::cout << "  INVISCID + BFECC clamp   n=" << cfg.lfm_cycle_steps << "  CG=" << cfg.solve_iters
              << "  dt=" << cfg.dt << " (CFL~" << cfg.dt / dx << "*Umax)\n";
    std::cout << "  cycles=" << n_cycles << "  t_end~" << n_cycles * cfg.lfm_cycle_steps * cfg.dt
              << "  out=" << cfg.out_dir << "/\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    CudaLFMSimulator3D sim(cfg);

    // Two coaxial rings on the x-axis with opposite circulation → head-on.
    scenarios::VortexRing left;
    left.center      = {0.35 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
    left.axis        = {1.0, 0.0, 0.0};
    left.radius      = 0.10; // in the y-z plane (spans 1.0) → room to expand ~4x
    left.core        = 0.022;
    left.circulation = +1.0;
    left.n_segments  = 300;
    scenarios::VortexRing right = left;
    right.center                = {0.65 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
    right.circulation           = -1.0;
    scenarios::add_vortex_ring(sim.mutable_grid(), left);
    scenarios::add_vortex_ring(sim.mutable_grid(), right);
    sim.commit();

    int frame = 0;
    write_vort_vtk(sim.grid(), frame++, cfg.out_dir);
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
    return 0;
}

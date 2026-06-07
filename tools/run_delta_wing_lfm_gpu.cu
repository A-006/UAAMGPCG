/**
 * @file run_delta_wing_lfm_gpu.cu
 * @brief Delta wing via GPU-resident LFM (paper Fig. 9) at paper resolution.
 *
 * Phase 2: full 256x128x128 (domain 2x1x1, dx uniform) on the GPU, using the
 * new freestream-box velocity BC + no-slip immersed solid, inviscid + BFECC
 * clamp + n=5. Sheds the leading-edge / wingtip vortices the coarse CPU run
 * couldn't resolve.
 *
 * Usage: run_delta_wing_lfm_gpu [cycles] [NX] [dt] [out_dir]
 *   defaults: cycles=400 NX=256 dt=2e-3 out=output_delta_wing_gpu
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "io/vtk_writer_3d.h"
#include "numerics/ops/operators_3d.h"
#include "simulator/cuda_lfm_simulator_3d.h"
#include "simulator/scenarios/3d/delta_wing.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

static void write_vort_vtk(const Grid3D& g, int frame, const std::string& dir) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/frame_%05d.vtk", dir.c_str(), frame);
    std::ofstream f(path);
    f << "# vtk DataFile Version 2.0\nLFM delta wing |w| - Frame " << frame
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
    cfg.NX  = 256; // domain 2x1x1 (paper aspect), dx=dy=dz=1/128
    cfg.NY  = 128;
    cfg.NZ  = 128;
    cfg.Lx  = 2.0;
    cfg.Ly  = 1.0;
    cfg.Lz  = 1.0;
    cfg.U_inf           = 0.6;
    cfg.Re              = 0;
    cfg.lfm_bfecc_clamp = true;
    cfg.dt              = 2e-3;
    cfg.solve_iters     = 200; // bigger frontal blockage (thin plate ⊥ AoA) needs more CG
    cfg.solve_tol       = 1e-6;
    cfg.time_integrator = "lfm";
    cfg.lfm_cycle_steps = 5;
    cfg.lfm_bc          = "freestream";
    cfg.out_dir         = "output_delta_wing_gpu";

    int n_cycles = 400;
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
    std::string sdf_path; // optional: authors' exact wing geometry (.npy SDF)
    if (argc > 5)
        sdf_path = argv[5];
    int frame_skip = 4;

    double aoa = 20.0 * M_PI / 180.0;
    cfg.inflow_ux = cfg.U_inf * std::cos(aoa); // 0.5638
    cfg.inflow_uy = cfg.U_inf * std::sin(aoa); // 0.2052
    cfg.inflow_uz = 0.0;

    double dx = cfg.Lx / cfg.NX;
    std::cout << "=== GPU LFM Delta Wing (Phase 2, paper res) ===\n";
    std::cout << "  Grid " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ << "  dx=" << dx
              << "  INVISCID+clamp  n=" << cfg.lfm_cycle_steps << "  freestream |U|=" << cfg.U_inf
              << "@20deg  dt=" << cfg.dt << " (CFL~" << cfg.U_inf * cfg.dt / dx << ")\n";
    std::cout << "  cycles=" << n_cycles << "  CG=" << cfg.solve_iters << "  out=" << cfg.out_dir
              << "/\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);
    CudaLFMSimulator3D sim(cfg);

    if (!sdf_path.empty()) {
        // Authors' exact wing geometry (cell-for-cell): mark sdf<0 as solid.
        scenarios::load_sdf_solid(sim.mutable_grid(), sdf_path);
    } else {
        scenarios::DeltaWing wing;
        wing.leading_x = 0.5;
        wing.chord     = 1.0;
        wing.semi_span = 0.35;
        wing.thickness = 0.02;
        wing.aoa_deg   = 20.0;
        wing.y_mid     = 0.5;
        scenarios::setup_delta_wing(sim.mutable_grid(), wing);
    }
    scenarios::set_uniform_freestream(sim.mutable_grid(), cfg.inflow_ux, cfg.inflow_uy, 0.0);
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

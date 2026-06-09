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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

// Load a shared staggered IC (ic{x,y,z}.raw, float32) into the grid's MAC faces —
// the SAME layout dump_collision_ic.cpp writes (icx[ix,iy,iz]=u_at(ix,iy+1,iz+1)).
// Lets us run our solver on the EXACT field the author's reference run loaded, so
// the cross-check is apples-to-apples at the initial condition.
static bool load_raw_ic(Grid3D& g, const std::string& dir) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    auto rd = [](const std::string& p, long n) {
        std::vector<float> buf(n);
        std::ifstream f(p, std::ios::binary);
        if (!f)
            return std::vector<float>();
        f.read((char*)buf.data(), n * sizeof(float));
        return buf;
    };
    auto bx = rd(dir + "/icx.raw", (long)(nx + 1) * ny * nz);
    auto by = rd(dir + "/icy.raw", (long)nx * (ny + 1) * nz);
    auto bz = rd(dir + "/icz.raw", (long)nx * ny * (nz + 1));
    if (bx.empty() || by.empty() || bz.empty())
        return false;
    long c = 0;
    for (int ix = 0; ix <= nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz < nz; iz++)
                g.u_at(ix, iy + 1, iz + 1) = bx[c++];
    c = 0;
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy <= ny; iy++)
            for (int iz = 0; iz < nz; iz++)
                g.v_at(ix + 1, iy, iz + 1) = by[c++];
    c = 0;
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz <= nz; iz++)
                g.w_at(ix + 1, iy + 1, iz) = bz[c++];
    return true;
}

// Dump cell-centered velocity (float32, C-order i-slowest, nx*ny*nz) so vorticity can
// be computed with the SAME np.gradient operator as the author's vx_*.npy — the only
// apples-to-apples way to compare |omega| trajectories across the two codes.
static void write_vel_raw(const Grid3D& g, int frame, const std::string& dir) {
    long n = (long)g.nx * g.ny * g.nz;
    std::vector<float> ux(n), uy(n), uz(n);
    long c = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            for (int k = 1; k <= g.nz; k++) {
                ux[c] = (float)(0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k)));
                uy[c] = (float)(0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k)));
                uz[c] = (float)(0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1)));
                c++;
            }
    char p[512];
    auto wr = [&](const char* nm, const std::vector<float>& b) {
        std::snprintf(p, sizeof(p), "%s/%s_%05d.raw", dir.c_str(), nm, frame);
        std::ofstream f(p, std::ios::binary);
        f.write((const char*)b.data(), b.size() * sizeof(float));
    };
    wr("vx", ux);
    wr("vy", uy);
    wr("vz", uz);
}

int main(int argc, char** argv) {
    Config cfg;
    cfg.dim = 3;
    // Paper aspect: collision axis x is the SHORT NX; rings expand into 2NX×2NX.
    // COLL_NX env overrides resolution for convergence studies (physical rings
    // via add_vortex_ring auto-scale, so finer grids resolve the core better).
    int base_nx = 128;
    if (const char* e = std::getenv("COLL_NX"))
        base_nx = std::atoi(e);
    cfg.NX = base_nx;
    cfg.NY = 2 * base_nx;
    cfg.NZ = 2 * base_nx;
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
    const char* ic_dir = std::getenv("COLL_IC_DIR");
    if (ic_dir && load_raw_ic(sim.mutable_grid(), ic_dir)) {
        std::printf("  loaded shared IC from %s/ (author cross-check field)\n", ic_dir);
    } else {
        scenarios::add_vortex_ring(sim.mutable_grid(), left);
        scenarios::add_vortex_ring(sim.mutable_grid(), right);
    }

    // ── one-shot IC diagnostic: max|u| and real fvc max|omega|, pre/post commit ──
    {
        auto diag = [](const Grid3D& g, const char* tag) {
            double umax = 0, wmax = 0;
            for (int k = 1; k <= g.nz; k++)
                for (int j = 1; j <= g.ny; j++)
                    for (int i = 1; i <= g.nx; i++) {
                        double uc = 0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k));
                        double vc = 0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k));
                        double wc = 0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1));
                        umax = std::max(umax, std::sqrt(uc * uc + vc * vc + wc * wc));
                        wmax = std::max(wmax, fvc::vorticity_magnitude(g, i, j, k));
                    }
            std::printf("  [diag %-12s] max|u|=%.4f  fvc max|omega|=%.2f\n", tag, umax, wmax);
        };
        diag(sim.grid(), "pre-commit");
        sim.commit();
        diag(sim.grid(), "post-commit");
    }

    bool dump_vel = std::getenv("DUMP_VEL") != nullptr;
    int frame = 0;
    write_vort_vtk(sim.grid(), frame, cfg.out_dir);
    if (dump_vel)
        write_vel_raw(sim.grid(), frame, cfg.out_dir);
    frame++;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int c = 1; c <= n_cycles; c++) {
        sim.step();
        if (c % frame_skip == 0 || c == n_cycles) {
            write_vort_vtk(sim.grid(), frame, cfg.out_dir);
            if (dump_vel)
                write_vel_raw(sim.grid(), frame, cfg.out_dir);
            frame++;
        }
        if (c % 5 == 0 || c == n_cycles)
            VtkWriter3D::printStatus(c, sim.time(), sim.grid());
    }
    auto t1   = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << n_cycles << " cycles in " << std::fixed << std::setprecision(1)
              << el << " s  (" << el / n_cycles << " s/cycle)\n";
    return 0;
}

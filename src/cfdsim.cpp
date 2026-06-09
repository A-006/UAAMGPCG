/**
 * @file src/cfdsim.cpp
 * @brief Unified, INPUT-file-driven launcher for all 3D LFM scenes (GPU or CPU).
 *
 * Single program for the whole project: built with CUDA it runs the GPU-resident
 * simulator (auto-detecting a device, CPU fallback if none); built without CUDA
 * it is a pure-C++ CPU binary. Compiled as CUDA (nvcc) or C++ (g++) from the SAME
 * source via a -DHAVE_CUDA switch — see CMakeLists.txt.
 *
 * One executable replaces the per-scene runners (run_vortex_ring_*,
 * run_vortex_ring_collision_*, run_delta_wing_*, run_collision_paper). The
 * scene and every parameter are read from an INI-style input file (and/or
 * `key=value` command-line overrides), so you never have to remember which
 * binary or which positional-argument order a scene wants.
 *
 *   ./cfdsim inputs/vortex_collision.in
 *   ./cfdsim inputs/vortex_collision.in NX=192 Re=300   # CLI overrides file
 *   ./cfdsim scenario=vortex_ring NX=96 cycles=80       # no file, all CLI
 *   ./cfdsim inputs/vortex_ring.in backend=cpu          # run on the CPU instead
 *   ./cfdsim scenario=karman NX=128 solver=pcg          # 2D (CPU) — see below
 *
 * 2D scenarios (karman, smoke, …) are dispatched to the shared 2D CPU pipeline
 * (the former lfm_2d); everything else is a 3D scenario run by this launcher.
 *
 * Supported scenarios (set `scenario=`):
 *   vortex_ring         single Gaussian-cored ring, propagates along +z
 *   vortex_collision    two opposite rings collide head-on (cube domain)
 *   collision_paper     paper Fig. 3 recipe (128x256x256, inviscid + BFECC clamp)
 *   delta_wing          immersed delta wing in a freestream box (paper Fig. 9)
 *   vortex_reconnection two tilted same-sign rings reconnect (paper Fig. 5)
 *   trefoil_knot        trefoil-knot vortex filament relaxes/breaks (paper Fig. 7)
 *
 * Common knobs (defaults are per-scene; see apply_scene_presets / the .in files):
 *   backend            "auto" (default: GPU if this build has CUDA and a device is
 *                      present, else CPU) | "gpu" (falls back to CPU if no GPU) |
 *                      "cpu" (host LFMSimulator3D; correct but far slower)
 *   NX NY NZ            grid resolution
 *   cycles             number of LFM reinitialization cycles to run
 *   dt                 time step
 *   Re                 Reynolds number (0 = inviscid)
 *   frame_skip         write a VTK frame every N cycles
 *   out_dir            output directory
 *   gpu                CUDA device index to run on (multi-GPU; backend=gpu only)
 *   solver             CPU-only pressure solver (cg | pcg | pcg_uaamg | ...; cg is
 *                      the robust 3D default. GPU backend uses its own built-in CG.)
 *   vtk_mode           "slim" (|omega| only, disk-friendly) or "full" (velocity
 *                      vectors + scalars via VtkWriter3D)
 * Scene-specific knobs: circulation, perturb_n, perturb_amp (rings); aoa_deg,
 *   sdf_path (delta wing); tilt_deg (reconnection); tk_scale (trefoil).
 * Diagnostics: ic_dir (load author's raw staggered IC), dump_vel (write
 *   cell-centered velocity .raw per frame) — for the author cross-check.
 */
#include "config/cli.h"
#include "config/config.h"
#include "core/grid_3d.h"
#include "io/vtk_writer_3d.h"
#include "numerics/bc/patches_3d.h"
#include "numerics/ops/operators_3d.h"
#include "simulator/factory.h" // 2D path: SimulatorFactory::create
#include "simulator/lfm_simulator_3d.h"
#include "simulator/runner.h" // 2D path: sim::run
#include "simulator/scenarios/3d/delta_wing.h"
#include "simulator/scenarios/3d/trefoil_knot.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include "simulator/simulator_3d.h"
#include "solver/factory_3d.h"
// CUDA is OPTIONAL. When built with nvcc (HAVE_CUDA) the GPU-resident simulator
// is available; otherwise cfdsim is a pure-C++ binary with the CPU backend only.
#ifdef HAVE_CUDA
#include "simulator/cuda_lfm_simulator_3d.h"
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

// ── Slim VTK: vorticity-magnitude scalar only (structured points) ──────────
// At high resolution the full VtkWriter3D (velocity vectors + extra scalars) is
// ~0.5–4.5 GB/frame and fills the disk; the iso-surface / volume renderers only
// need |omega|. This is the same format the per-scene runners wrote.
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

// ── Author cross-check diagnostics (ported from run_collision_paper) ────────
// Load a shared staggered IC (ic{x,y,z}.raw, float32) into the grid's MAC faces
// — the SAME layout dump_collision_ic writes. Lets us run our solver on the EXACT
// field the author's reference loaded (apples-to-apples at the initial step).
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

// Dump cell-centered velocity (float32, C-order i-slowest, nx*ny*nz) so |omega|
// can be computed with the SAME np.gradient operator as the author's vx_*.npy —
// the only apples-to-apples way to compare |omega| trajectories across codes.
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

// ── Per-scene defaults ─────────────────────────────────────────────────────
// Fill cfg with a scene's baseline; the caller then re-applies the INI/CLI
// assignments on top so any field the user wrote wins. Scene-specific knobs go
// into cfg.extra as string defaults (so an explicit override replaces them).
static void apply_scene_presets(Config& cfg) {
    cfg.dim             = 3;
    cfg.time_integrator = "lfm";
    cfg.solver          = "cg"; // CPU backend: robust 3D choice (GPU has its own CG)
    cfg.extra["backend"] = "auto"; // GPU if available, else CPU (see main)
    const std::string& s = cfg.scenario;

    if (s == "vortex_ring") {
        cfg.NX = cfg.NY = cfg.NZ = 64;
        cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
        cfg.U_inf = 1.0;
        cfg.Re    = 0;
        cfg.cyl_R = 0.1;
        cfg.dt              = 0.002;
        cfg.solve_iters     = 200;
        cfg.solve_tol       = 1e-6;
        cfg.lfm_cycle_steps = 2;
        cfg.frame_skip      = 1;
        cfg.out_dir         = "output_vortex_ring";
        cfg.extra["cycles"]      = "60";
        cfg.extra["circulation"] = "1.0";
        cfg.extra["vtk_mode"]    = "full"; // small grid → nice full-field viz
    } else if (s == "vortex_collision") {
        cfg.NX = cfg.NY = cfg.NZ = 96;
        cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
        cfg.U_inf = 1.0;
        cfg.Re    = 0;
        cfg.cyl_R = 0.1;
        cfg.dt              = 0.001;
        cfg.solve_iters     = 120;
        cfg.solve_tol       = 1e-6;
        cfg.lfm_cycle_steps = 2;
        cfg.frame_skip      = 4;
        cfg.out_dir         = "output_vortex_collision";
        cfg.extra["cycles"]      = "120";
        cfg.extra["circulation"] = "1.0";
        cfg.extra["perturb_n"]   = "0";
        cfg.extra["perturb_amp"] = "0.0";
        cfg.extra["vtk_mode"]    = "slim";
    } else if (s == "collision_paper") {
        // Paper aspect: collision axis x is the SHORT NX; rings expand into the
        // large 2NX x 2NX plane. Domain 0.5x1x1 → dx=dy=dz uniform (= 1/256).
        cfg.NX = 128;
        cfg.NY = 256;
        cfg.NZ = 256;
        cfg.Lx = 0.5;
        cfg.Ly = 1.0;
        cfg.Lz = 1.0;
        cfg.U_inf           = 1.0;
        cfg.Re              = 0;    // inviscid — stability comes from BFECC clamp
        cfg.lfm_bfecc_clamp = true; // paper's BfeccClamp (numerical viscosity)
        cfg.cyl_R           = 0.1;
        cfg.dt              = 4e-4;
        cfg.solve_iters     = 8; // paper: CG fixed at 8 iterations
        cfg.solve_tol       = 0.0;
        cfg.lfm_cycle_steps = 5; // paper: n = 5 steps per reinitialization cycle
        cfg.frame_skip      = 4;
        cfg.out_dir         = "output_collision_paper";
        cfg.extra["cycles"]   = "250";
        cfg.extra["vtk_mode"] = "slim";
    } else if (s == "delta_wing") {
        cfg.NX = 256; // domain 2x1x1 (paper aspect), dx=dy=dz=1/128
        cfg.NY = 128;
        cfg.NZ = 128;
        cfg.Lx = 2.0;
        cfg.Ly = 1.0;
        cfg.Lz = 1.0;
        cfg.U_inf           = 0.6;
        cfg.Re              = 0;
        cfg.lfm_bfecc_clamp = true;
        cfg.dt              = 2e-3;
        cfg.solve_iters     = 200;
        cfg.solve_tol       = 1e-6;
        cfg.lfm_cycle_steps = 5;
        cfg.lfm_bc          = "freestream";
        cfg.frame_skip      = 4;
        cfg.out_dir         = "output_delta_wing";
        cfg.extra["cycles"]   = "400";
        cfg.extra["aoa_deg"]  = "20";
        cfg.extra["sdf_path"] = "";
        cfg.extra["vtk_mode"] = "slim";
    } else if (s == "vortex_reconnection") {
        // Two same-sign rings, offset + tilted 25° so their near sides approach
        // and reconnect (paper Fig. 5). Reconnection is viscosity-driven → Re>0.
        cfg.NX = cfg.NY = cfg.NZ = 64;
        cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
        cfg.U_inf = 1.0;
        cfg.Re    = 2000.0;
        cfg.cyl_R = 0.1;
        cfg.dt              = 0.004;
        cfg.solve_iters     = 200;
        cfg.solve_tol       = 1e-8;
        cfg.lfm_cycle_steps = 2;
        cfg.frame_skip      = 4;
        cfg.out_dir         = "output_vortex_reconnection";
        cfg.extra["cycles"]      = "200";
        cfg.extra["circulation"] = "0.6";
        cfg.extra["tilt_deg"]    = "25";
        cfg.extra["vtk_mode"]    = "full";
    } else if (s == "trefoil_knot") {
        // Trefoil-knot vortex filament (paper Fig. 7): relaxes and breaks into a
        // large + small vortex. Viscous (Re>0).
        cfg.NX = cfg.NY = cfg.NZ = 64;
        cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
        cfg.U_inf = 1.0;
        cfg.Re    = 2000.0;
        cfg.cyl_R = 0.05;
        cfg.dt              = 0.004;
        cfg.solve_iters     = 200;
        cfg.solve_tol       = 1e-8;
        cfg.lfm_cycle_steps = 2;
        cfg.frame_skip      = 5;
        cfg.out_dir         = "output_trefoil";
        cfg.extra["cycles"]      = "250";
        cfg.extra["circulation"] = "0.5";
        cfg.extra["tk_scale"]    = "0.07";
        cfg.extra["vtk_mode"]    = "full";
    } else {
        throw std::runtime_error(
            "cfdsim: unknown scenario '" + s +
            "'; known: vortex_ring | vortex_collision | collision_paper | delta_wing | "
            "vortex_reconnection | trefoil_knot");
    }
}

// ── Per-scene initial condition (written onto the host grid) ────────────────
// Backend-agnostic: operates on a plain Grid3D, so the GPU and CPU simulators
// share the exact same initial condition.
static void setup_scene(Grid3D& g, const Config& cfg) {
    const std::string& s = cfg.scenario;

    // Author cross-check: if ic_dir is set, load that exact staggered field and
    // skip the analytic IC (collision_paper apples-to-apples comparison).
    std::string ic_dir = cfg.sget("ic_dir", "");
    if (!ic_dir.empty()) {
        if (load_raw_ic(g, ic_dir)) {
            std::printf("  loaded shared IC from %s/ (author cross-check field)\n", ic_dir.c_str());
            return;
        }
        std::fprintf(stderr, "  WARNING: ic_dir='%s' unreadable — falling back to analytic IC\n",
                     ic_dir.c_str());
    }

    if (s == "vortex_ring") {
        scenarios::VortexRing ring;
        ring.center      = {0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.28 * cfg.Lz};
        ring.axis        = {0.0, 0.0, 1.0};
        ring.radius      = 0.18 * cfg.Lx;
        ring.core        = 0.045 * cfg.Lx;
        ring.circulation = cfg.dget("circulation", 1.0);
        ring.n_segments  = 240;
        scenarios::add_vortex_ring(g, ring);
    } else if (s == "vortex_collision") {
        double circ = cfg.dget("circulation", 1.0);
        scenarios::VortexRing left;
        left.center      = {0.36 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        left.axis        = {1.0, 0.0, 0.0};
        left.radius      = 0.10 * cfg.Lx;
        left.core        = 0.025 * cfg.Lx;
        left.circulation = +circ;
        left.n_segments  = 200;
        left.perturb_n   = cfg.iget("perturb_n", 0);
        left.perturb_amp = cfg.dget("perturb_amp", 0.0);
        scenarios::VortexRing right = left;
        right.center                = {0.64 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        right.circulation           = -circ;
        scenarios::add_vortex_ring(g, left);
        scenarios::add_vortex_ring(g, right);
    } else if (s == "collision_paper") {
        // Physical rings in the y-z plane (spans 1.0) → room to expand ~4x.
        scenarios::VortexRing left;
        left.center      = {0.35 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        left.axis        = {1.0, 0.0, 0.0};
        left.radius      = 0.10;
        left.core        = 0.022;
        left.circulation = +1.0;
        left.n_segments  = 300;
        scenarios::VortexRing right = left;
        right.center                = {0.65 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        right.circulation           = -1.0;
        scenarios::add_vortex_ring(g, left);
        scenarios::add_vortex_ring(g, right);
    } else if (s == "delta_wing") {
        std::string sdf = cfg.sget("sdf_path", "");
        if (!sdf.empty()) {
            scenarios::load_sdf_solid(g, sdf); // authors' exact wing geometry
        } else {
            scenarios::DeltaWing wing;
            wing.leading_x = 0.5;
            wing.chord     = 1.0;
            wing.semi_span = 0.35;
            wing.thickness = 0.02;
            wing.aoa_deg   = cfg.dget("aoa_deg", 20.0);
            wing.y_mid     = 0.5;
            scenarios::setup_delta_wing(g, wing);
        }
        scenarios::set_uniform_freestream(g, cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
    } else if (s == "vortex_reconnection") {
        double circ = cfg.dget("circulation", 0.6);
        double tilt = cfg.dget("tilt_deg", 25.0) * M_PI / 180.0;
        double ct = std::cos(tilt), st = std::sin(tilt);
        scenarios::VortexRing ringA = {{0.40 * cfg.Lx, 0.50 * cfg.Ly, 0.50 * cfg.Lz},
                                       {ct, +st, 0.0}, 0.14, 0.025, +circ, 240};
        scenarios::VortexRing ringB = {{0.60 * cfg.Lx, 0.50 * cfg.Ly, 0.50 * cfg.Lz},
                                       {ct, -st, 0.0}, 0.14, 0.025, +circ, 240};
        scenarios::add_vortex_ring(g, ringA);
        scenarios::add_vortex_ring(g, ringB);
    } else if (s == "trefoil_knot") {
        scenarios::TrefoilKnot tk{};
        tk.center      = {0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        tk.scale       = cfg.dget("tk_scale", 0.07);
        tk.core        = 0.025;
        tk.circulation = cfg.dget("circulation", 0.5);
        tk.n_segments  = 360;
        scenarios::add_trefoil_knot(g, tk);
    }
}

// ── Wall boundary conditions for the CPU backend ───────────────────────────
// The GPU CudaLFMSimulator3D applies these internally in commit(); the host
// LFMSimulator3D needs an explicit BoundaryManager3D. Mirror the same choice:
// freestream box for the delta wing, free-slip box (+ immersed solid) otherwise.
static bc::BoundaryManager3D make_cpu_bcs(const Config& cfg) {
    if (cfg.lfm_bc == "freestream")
        return scenarios::freestream_box_bcs(cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
    return bc::free_slip_box();
}

// The 3D scenarios this launcher provisions itself. Anything else (karman,
// smoke, …) is a 2D scenario handled by the shared 2D pipeline below.
static const std::set<std::string> k3DScenarios = {
    "vortex_ring",  "vortex_collision",    "collision_paper",
    "delta_wing",   "vortex_reconnection", "trefoil_knot"};

// 2D path — identical to the former lfm_2d/main.cpp: the 2D ScenarioRegistry
// configures the run and sim::run drives it (CPU; the 2D solvers are host-only).
static int run_2d(int argc, char** argv) {
    auto cfg = config::parse_cli(argc, argv);
    if (!cfg)
        return 1;
    auto sim = SimulatorFactory::create(*cfg);
    sim::run(*sim, *cfg);
    return 0;
}

int main(int argc, char** argv) {
    // Dispatch 2D vs 3D by the (peeked) scenario name. 2D scenarios go through the
    // shared CPU pipeline; the 3D scenarios below are GPU/CPU via this launcher.
    {
        std::string scenario = "vortex_ring"; // default
        try {
            for (const auto& [k, v] : config::collect_assignments(argc, argv))
                if (k == "scenario")
                    scenario = v;
        } catch (...) {
            // fall through; the chosen path reports the error properly
        }
        if (k3DScenarios.count(scenario) == 0)
            return run_2d(argc, argv); // karman / smoke / … (2D)
    }

    // 1) Collect file + CLI assignments, peek the scenario, apply its presets,
    //    then re-apply the assignments so any user override wins.
    config::KeyVals kv;
    Config cfg;
    try {
        kv = config::collect_assignments(argc, argv);
        cfg.scenario = "vortex_ring"; // default if none given
        for (const auto& [k, v] : kv)
            if (k == "scenario")
                cfg.scenario = v;
        apply_scene_presets(cfg);
        for (const auto& [k, v] : kv)
            config::set_field(cfg, k, v);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"
                  << "Usage: cfdsim [input.in] [key=value]...\n"
                     "  scenarios: vortex_ring | vortex_collision | collision_paper | delta_wing | "
                     "vortex_reconnection | trefoil_knot\n";
        return 1;
    }

    // 2) Delta wing: derive the freestream from |U|@angle-of-attack unless the
    //    user set inflow_* explicitly.
    if (cfg.scenario == "delta_wing" && cfg.inflow_ux == 0.0 && cfg.inflow_uy == 0.0 &&
        cfg.inflow_uz == 0.0) {
        double aoa    = cfg.dget("aoa_deg", 20.0) * M_PI / 180.0;
        cfg.inflow_ux = cfg.U_inf * std::cos(aoa);
        cfg.inflow_uy = cfg.U_inf * std::sin(aoa);
    }

    // 3) Choose the compute backend: "auto" (default) | "gpu" | "cpu".
    //    auto  → GPU if this build has CUDA *and* a device is present, else CPU.
    //    gpu   → GPU; if unavailable, warn and fall back to CPU (never hard-fail).
    //    cpu   → always the host LFMSimulator3D.
    std::string backend = cfg.sget("backend", "auto");
    if (backend != "auto" && backend != "gpu" && backend != "cpu") {
        std::cerr << "cfdsim: backend must be 'auto' | 'gpu' | 'cpu' (got '" << backend << "')\n";
        return 1;
    }

    // Is a usable GPU actually available in THIS build / on THIS machine?
    bool gpu_available = false;
#ifdef HAVE_CUDA
    {
        int ndev = 0;
        gpu_available = (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0);
    }
#endif

    if (backend == "auto")
        backend = gpu_available ? "gpu" : "cpu";
    if (backend == "gpu" && !gpu_available) {
#ifdef HAVE_CUDA
        std::cerr << "cfdsim: no CUDA device found — falling back to backend=cpu\n";
#else
        std::cerr << "cfdsim: built without CUDA — falling back to backend=cpu\n";
#endif
        backend = "cpu";
    }

    int dev = cfg.iget("gpu", -1); // multi-GPU device index (backend=gpu only)
#ifdef HAVE_CUDA
    if (backend == "gpu" && dev >= 0) {
        if (cudaSetDevice(dev) != cudaSuccess) {
            std::cerr << "cfdsim: cudaSetDevice(" << dev << ") failed\n";
            return 1;
        }
    }
#endif

    int n_cycles         = cfg.iget("cycles", 100);
    std::string vtk_mode = cfg.sget("vtk_mode", "slim");
    bool full_vtk        = (vtk_mode == "full");
    bool dump_vel        = cfg.iget("dump_vel", 0) != 0; // cell-centered velocity .raw per frame
    double dx            = cfg.Lx / cfg.NX;

    std::cout << "===================================================\n";
    std::cout << "  3D LFM (" << backend << ") — scenario: " << cfg.scenario << "\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ << "  dx=" << dx << "\n";
    std::cout << "  dt=" << cfg.dt << "  n=" << cfg.lfm_cycle_steps << "/cycle  cycles=" << n_cycles
              << "  Re=" << cfg.Re << "  clamp=" << (cfg.lfm_bfecc_clamp ? "on" : "off") << "\n";
    std::cout << "  bc=" << cfg.lfm_bc << "  vtk=" << vtk_mode << "  out=" << cfg.out_dir << "/"
              << (backend == "gpu" ? (dev >= 0 ? ("  gpu=" + std::to_string(dev)) : "")
                                   : ("  solver=" + cfg.solver))
              << "\n";
    if (backend == "cpu")
        std::cout << "  [note] CPU backend is a correctness reference — much slower than GPU;"
                     " use small grids.\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);

    // Build the chosen simulator. Both derive from Simulator3D, so the output
    // loop below is identical; only construction + initial BC differ. The GPU
    // branch only exists in CUDA builds (guarded); by here backend=="gpu" implies
    // HAVE_CUDA, so the CPU-only build never reaches it.
    std::unique_ptr<Simulator3D> sim;
#ifdef HAVE_CUDA
    if (backend == "gpu") {
        auto s = std::make_unique<CudaLFMSimulator3D>(cfg);
        setup_scene(s->mutable_grid(), cfg);
        s->commit(); // upload IC + apply free-slip / freestream wall BC
        sim = std::move(s);
    } else
#endif
    {
        auto s = std::make_unique<LFMSimulator3D>(cfg, Factory3D::create(cfg.solver));
        setup_scene(s->mutable_grid(), cfg);
        s->set_boundary_manager(make_cpu_bcs(cfg)); // applies the wall BC
        sim = std::move(s);
    }

    auto write_frame = [&](int frame) {
        if (full_vtk)
            VtkWriter3D::write(sim->grid(), frame, cfg);
        else
            write_vort_vtk(sim->grid(), frame, cfg.out_dir);
        if (dump_vel)
            write_vel_raw(sim->grid(), frame, cfg.out_dir);
    };

    int frame = 0;
    write_frame(frame++);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int c = 1; c <= n_cycles; c++) {
        sim->step();
        if (c % cfg.frame_skip == 0 || c == n_cycles)
            write_frame(frame++);
        if (c % 5 == 0 || c == n_cycles)
            VtkWriter3D::printStatus(c, sim->time(), sim->grid());
    }
    auto t1   = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << n_cycles << " cycles in " << std::fixed << std::setprecision(1)
              << el << " s  (" << el / n_cycles << " s/cycle)\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";
    return 0;
}

#include "simulator/scene_3d.h"
#include "config/cli.h"
#include "simulator/scenarios/3d/delta_wing.h"
#include "simulator/scenarios/3d/trefoil_knot.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

namespace scene3d {

// The 3D scenarios this launcher provisions itself. Anything else (karman,
// smoke, …) is a 2D scenario handled by the shared 2D pipeline.
static const std::set<std::string> kScenarios = {
    "vortex_ring",  "vortex_collision",    "collision_paper",
    "delta_wing",   "vortex_reconnection", "trefoil_knot"};

bool is_3d_scenario(const std::string& name) {
    return kScenarios.count(name) != 0;
}

// ── Per-scene defaults ─────────────────────────────────────────────────────
// Fill cfg with a scene's baseline; the caller then re-applies the INI/CLI
// assignments on top so any field the user wrote wins. Scene-specific knobs go
// into cfg.extra as string defaults (so an explicit override replaces them).
static void apply_presets(Config& cfg) {
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

std::string peek_scenario(int argc, char** argv) {
    std::string scenario = "vortex_ring"; // default
    try {
        for (const auto& [k, v] : config::collect_assignments(argc, argv))
            if (k == "scenario")
                scenario = v;
    } catch (...) {
        // fall through; the chosen path reports the error properly
    }
    return scenario;
}

Config build_config(int argc, char** argv) {
    // Collect file + CLI assignments, peek the scenario, apply its presets, then
    // re-apply the assignments so any user override wins.
    config::KeyVals kv = config::collect_assignments(argc, argv);
    Config cfg;
    cfg.scenario = "vortex_ring"; // default if none given
    for (const auto& [k, v] : kv)
        if (k == "scenario")
            cfg.scenario = v;
    apply_presets(cfg);
    for (const auto& [k, v] : kv)
        config::set_field(cfg, k, v);

    // Delta wing: derive the freestream from |U|@angle-of-attack unless the user
    // set inflow_* explicitly.
    if (cfg.scenario == "delta_wing" && cfg.inflow_ux == 0.0 && cfg.inflow_uy == 0.0 &&
        cfg.inflow_uz == 0.0) {
        double aoa    = cfg.dget("aoa_deg", 20.0) * M_PI / 180.0;
        cfg.inflow_ux = cfg.U_inf * std::cos(aoa);
        cfg.inflow_uy = cfg.U_inf * std::sin(aoa);
    }
    return cfg;
}

// ── Author cross-check: load a shared staggered IC (ic{x,y,z}.raw, float32) ──
// into the grid's MAC faces — the SAME layout dump_collision_ic writes. Lets us
// run our solver on the EXACT field the author's reference loaded.
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

void setup(Grid3D& g, const Config& cfg) {
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

bc::BoundaryManager3D make_cpu_bcs(const Config& cfg) {
    if (cfg.lfm_bc == "freestream")
        return scenarios::freestream_box_bcs(cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
    return bc::free_slip_box();
}

} // namespace scene3d

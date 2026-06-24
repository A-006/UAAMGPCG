#include "simulator/scene_3d.h"
#include "config/cli.h"
#include "io/vtk_slim_3d.h"
#include "simulator/scenarios/3d/delta_wing.h"
#include "simulator/scenarios/3d/trefoil_knot.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace scene3d {

// The 3D scenarios this launcher provisions itself. Anything else (karman,
// smoke, …) is a 2D scenario handled by the shared 2D pipeline.
static const std::set<std::string> kScenarios = {
    "vortex_ring",  "vortex_collision",    "collision_paper", "leapfrog_rings",
    "delta_wing",   "vortex_reconnection", "trefoil_knot"};

bool is_3d_scenario(const std::string& name) {
    return kScenarios.count(name) != 0;
}

// Helpers used below — defined after the public entry points so this file reads
// top-down: the launcher's two calls first, the details underneath.
static void apply_presets(Config& cfg);
static std::string scenario_of(const config::KeyVals& kv);
static void derive_freestream(Config& cfg);

std::string peek_scenario(int argc, char** argv) {
    try {
        return scenario_of(config::collect_assignments(argc, argv));
    } catch (...) {
        return "vortex_ring"; // the chosen path reports the real error later
    }
}

Config build_config(int argc, char** argv) {
    config::KeyVals kv = config::collect_assignments(argc, argv);

    // Lay down the scenario's presets, then apply the user's assignments on top
    // so any explicit override wins.
    Config cfg;
    cfg.scenario = scenario_of(kv);
    apply_presets(cfg);
    for (const auto& [k, v] : kv)
        config::set_field(cfg, k, v);

    // Then fill in fields computed from the others, now that everything is set
    // (currently just the delta-wing freestream from its angle of attack).
    derive_freestream(cfg);
    return cfg;
}

// Directory holding the running executable (Linux /proc), or "" if unknown.
static std::string exe_dir() {
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        return "";
    buf[n]           = '\0';
    std::string path = buf;
    auto slash       = path.find_last_of('/');
    return slash == std::string::npos ? "" : path.substr(0, slash);
}

// Locate a scenario's preset file inputs/<scenario>.in. Search order:
// $UAAMG_PRESETS_DIR, the inputs/ dir beside (or one level above) the
// executable, then ./inputs. Returns "" if none of them has the file.
static std::string find_preset(const std::string& scenario) {
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("UAAMG_PRESETS_DIR"))
        dirs.push_back(env);
    if (std::string exe = exe_dir(); !exe.empty()) {
        dirs.push_back(exe + "/../inputs");
        dirs.push_back(exe + "/inputs");
    }
    dirs.push_back("inputs");

    for (const auto& dir : dirs) {
        std::string path = dir + "/" + scenario + ".in";
        if (std::ifstream(path).good())
            return path;
    }
    return "";
}

// ── Per-scene defaults ─────────────────────────────────────────────────────
// Every 3D scenario shares the LFM base below; the rest of the recipe is DATA,
// read from its preset file inputs/<scenario>.in (so grid size, dt, cycles, …
// change with no rebuild). build_config then layers the user's INI/CLI
// assignments on top so any explicit override still wins.
static void apply_presets(Config& cfg) {
    cfg.dim             = 3;
    cfg.time_integrator = "lfm";
    cfg.solver          = "cg"; // CPU backend: robust 3D choice (GPU has its own CG)
    cfg.extra["backend"] = "auto"; // GPU if available, else CPU (see main)

    std::string path = find_preset(cfg.scenario);
    if (path.empty())
        throw std::runtime_error("cfdsim: no preset for scenario '" + cfg.scenario +
                                 "'; expected inputs/" + cfg.scenario +
                                 ".in (or set UAAMG_PRESETS_DIR)");
    for (const auto& [key, value] : config::read_assignments(path))
        config::set_field(cfg, key, value);
}

// Which scenario do these assignments select? Last `scenario=` wins; the
// default applies when none is given.
static std::string scenario_of(const config::KeyVals& kv) {
    std::string scenario = "vortex_ring";
    for (const auto& [k, v] : kv)
        if (k == "scenario")
            scenario = v;
    return scenario;
}

// Delta wing: derive the freestream from |U| at the angle of attack, unless the
// user set inflow_* explicitly.
static void derive_freestream(Config& cfg) {
    bool inflow_set = cfg.inflow_ux != 0.0 || cfg.inflow_uy != 0.0 || cfg.inflow_uz != 0.0;
    if (cfg.scenario != "delta_wing" || inflow_set)
        return;
    double aoa    = cfg.dget("aoa_deg", 20.0) * M_PI / 180.0;
    cfg.inflow_ux = cfg.U_inf * std::cos(aoa);
    cfg.inflow_uy = cfg.U_inf * std::sin(aoa);
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
    } else if (s == "leapfrog_rings") {
        // Two coaxial, same-sign rings offset along x. The trailing ring is
        // induced to contract and slip through the leading one, then they swap
        // roles — the classic leapfrog (paper Fig. 14).
        double circ = cfg.dget("circulation", 1.0);
        scenarios::VortexRing lead;
        lead.center      = {0.45 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        lead.axis        = {1.0, 0.0, 0.0};
        lead.radius      = 0.18 * cfg.Ly;
        lead.core        = 0.04 * cfg.Ly;
        lead.circulation = +circ;
        lead.n_segments  = 240;
        scenarios::VortexRing trail = lead;
        trail.center                = {0.30 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz};
        scenarios::add_vortex_ring(g, lead);
        scenarios::add_vortex_ring(g, trail);
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

    // Author cross-check export: dump the EXACT analytic IC our solver will run
    // (MAC faces) so the reference runner can load the identical field. Wrap the
    // ic{x,y,z}.raw into init_u_{x,y,z}.npy for any scenario, not just collision.
    std::string dump_ic = cfg.sget("dump_ic_dir", "");
    if (!dump_ic.empty()) {
        io3d::write_face_ic(g, dump_ic);
        std::printf("  dumped analytic face IC to %s/ic{x,y,z}.raw\n", dump_ic.c_str());
    }
}

bc::BoundaryManager3D make_cpu_bcs(const Config& cfg) {
    if (cfg.lfm_bc == "freestream")
        return scenarios::freestream_box_bcs(cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
    return bc::free_slip_box();
}

} // namespace scene3d

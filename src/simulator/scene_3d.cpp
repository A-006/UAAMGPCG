#include "simulator/scene_3d.h"
#include "config/cli.h"
#include "io/vtk_slim_3d.h"
#include "ic/3d/delta_wing.h"
#include "ic/3d/trefoil_knot.h"
#include "ic/3d/vortex_ring.h"
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace scene3d {

// Helpers used below — defined after the public entry points so this file reads
// top-down: the launcher's two calls first, the details underneath.
static void apply_presets(Config& cfg);
static std::string scenario_of(const config::KeyVals& kv);
static void derive_freestream(Config& cfg);
static std::string find_preset(const std::string& scenario);

std::string peek_scenario(int argc, char** argv) {
    try {
        return scenario_of(config::collect_assignments(argc, argv));
    } catch (...) {
        return "vortex_ring"; // the chosen path reports the real error later
    }
}

std::string peek_key(int argc, char** argv, const std::string& key, const std::string& def) {
    try {
        std::string out = def;
        for (const auto& [k, v] : config::collect_assignments(argc, argv))
            if (k == key)
                out = v;
        return out;
    } catch (...) {
        return def; // the chosen path reports the real error later
    }
}

std::string peek_dim(int argc, char** argv) {
    try {
        std::string dim, scenario;
        bool have_dim = false;
        for (const auto& [k, v] : config::collect_assignments(argc, argv)) {
            if (k == "dim") {
                dim      = v;
                have_dim = true;
            } else if (k == "scenario")
                scenario = v;
        }
        if (have_dim)
            return dim; // explicit dim in file/CLI wins
        // Otherwise take dim from the scenario's case file (where dim=3 lives).
        if (!scenario.empty())
            if (std::string path = find_preset(scenario); !path.empty())
                for (const auto& [k, v] : config::read_assignments(path))
                    if (k == "dim")
                        dim = v;
        if (!dim.empty())
            return dim;
    } catch (...) {
    }
    return "2";
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

// ── IC primitive registry ───────────────────────────────────────────────────
// The solver core has ZERO knowledge of scenario names. Each case file declares
// one or more IC SOURCES; every source picks a primitive by `kind` and reads its
// own geometry from the config. This decouples "what the IC is" (data, in
// inputs/<scenario>.in) from "how to build it" (the analytic math in
// scenarios/3d/*, which we DO NOT rewrite — only call).
//
// Case-file syntax (index N = 0,1,2,…):
//   ic    = vortex_ring     # kind of source 0 (alias for ic0)
//   ic1   = vortex_ring     # kind of source 1
//   ic0.center      = 0.4,0.5,0.5
//   ic0.circulation = 1.0
//   ic1.circulation = -1.0
// A source is "present" iff its kind key (`ic` for index 0, `icN` for N≥1) is
// set. Parameters of source N are read from `icN.<param>`; for source 0 the
// unindexed `ic.<param>` is accepted too (so a single-source case can write
// `ic = vortex_ring` + `ic.center = …`). Keys absent from the case fall back to
// the builder's documented default, so every preset stays fully data-driven.

// A reader bound to one IC source: looks up `<prefix>.<param>` (e.g. "ic0.core")
// in cfg.extra, with the same default-on-miss semantics as cfg.dget/iget/etc.
namespace {
struct IcParams {
    const Config* cfg;
    std::string prefix;  // "ic0", "ic1", … (the indexed prefix for this source)
    std::string prefix0; // "ic" for source 0, else "" (the unindexed alias)

    // Resolve a param: prefer the indexed key, fall back to the unindexed alias
    // (only meaningful for source 0), then to `def`.
    template <class T, class Get>
    T get(const std::string& name, T def, Get getter) const {
        std::string ik = prefix + "." + name;
        if (cfg->extra.count(ik))
            return getter(ik, def);
        if (!prefix0.empty() && cfg->extra.count(prefix0 + "." + name))
            return getter(prefix0 + "." + name, def);
        return def;
    }
    double d(const std::string& n, double def) const {
        return get(n, def, [&](const std::string& k, double dd) { return cfg->dget(k, dd); });
    }
    int i(const std::string& n, int def) const {
        return get(n, def, [&](const std::string& k, int dd) { return cfg->iget(k, dd); });
    }
    std::string s(const std::string& n, const std::string& def) const {
        return get(n, def,
                   [&](const std::string& k, const std::string& dd) { return cfg->sget(k, dd); });
    }
    std::array<double, 3> v3(const std::string& n, std::array<double, 3> def) const {
        return get(n, def, [&](const std::string& k, std::array<double, 3> dd) {
            return cfg->v3get(k, dd);
        });
    }
};
} // namespace

// A builder turns one source's params into geometry on the grid.
using IcBuilder = std::function<void(Grid3D&, const Config&, const IcParams&)>;

// kind "vortex_ring": Gaussian-cored ring via Biot-Savart. Defaults reproduce
// the single vortex_ring scenario; the multi-ring cases (collision, leapfrog,
// reconnection) set center/axis/radius/circulation per source.
static void build_vortex_ring(Grid3D& g, const Config& cfg, const IcParams& p) {
    scenarios::VortexRing vr;
    vr.center      = p.v3("center", {0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.28 * cfg.Lz});
    vr.axis        = p.v3("axis", {0.0, 0.0, 1.0});
    vr.radius      = p.d("radius", 0.18 * cfg.Lx);
    vr.core        = p.d("core", 0.045 * cfg.Lx);
    vr.circulation = p.d("circulation", 1.0);
    vr.n_segments  = p.i("n_segments", 240);
    vr.perturb_n   = p.i("perturb_n", 0);
    vr.perturb_amp = p.d("perturb_amp", 0.0);
    scenarios::add_vortex_ring(g, vr);
}

// kind "trefoil_knot": (2,3) torus-knot filament.
static void build_trefoil_knot(Grid3D& g, const Config& cfg, const IcParams& p) {
    scenarios::TrefoilKnot tk{};
    tk.center      = p.v3("center", {0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz});
    tk.scale       = p.d("scale", 0.07);
    tk.core        = p.d("core", 0.025);
    tk.circulation = p.d("circulation", 0.5);
    tk.n_segments  = p.i("n_segments", 360);
    scenarios::add_trefoil_knot(g, tk);
}

// kind "delta_wing": immersed solid (authors' SDF if sdf_path is set, else an
// analytic flat plate) + a uniform freestream IC from the derived inflow_*.
static void build_delta_wing(Grid3D& g, const Config& cfg, const IcParams& p) {
    std::string sdf = p.s("sdf_path", "");
    if (!sdf.empty()) {
        scenarios::load_sdf_solid(g, sdf); // authors' exact wing geometry
    } else {
        scenarios::DeltaWing wing;
        wing.leading_x = p.d("leading_x", 0.5);
        wing.chord     = p.d("chord", 1.0);
        wing.semi_span = p.d("semi_span", 0.35);
        wing.thickness = p.d("thickness", 0.02);
        wing.aoa_deg   = p.d("aoa_deg", cfg.dget("aoa_deg", 20.0));
        wing.y_mid     = p.d("y_mid", 0.5);
        scenarios::setup_delta_wing(g, wing);
    }
    scenarios::set_uniform_freestream(g, cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
}

static const std::unordered_map<std::string, IcBuilder>& ic_registry() {
    static const std::unordered_map<std::string, IcBuilder> kReg = {
        {"vortex_ring", build_vortex_ring},
        {"trefoil_knot", build_trefoil_knot},
        {"delta_wing", build_delta_wing},
    };
    return kReg;
}

void setup(Grid3D& g, const Config& cfg) {
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

    // Walk the IC sources the case declares: source 0 is `ic`, then ic1, ic2, …
    // Stop at the first index whose kind key is absent. Each present source picks
    // a primitive from the registry and applies it with its own params.
    int applied = 0;
    for (int n = 0;; n++) {
        std::string kind_key = (n == 0) ? "ic" : ("ic" + std::to_string(n));
        std::string kind     = cfg.sget(kind_key, "");
        if (kind.empty())
            break;
        auto it = ic_registry().find(kind);
        if (it == ic_registry().end())
            throw std::runtime_error("cfdsim: unknown IC kind '" + kind + "' for " + kind_key +
                                     " (see scene3d::ic_registry)");
        IcParams p{&cfg, "ic" + std::to_string(n), (n == 0) ? "ic" : ""};
        it->second(g, cfg, p);
        applied++;
    }
    if (applied == 0)
        std::fprintf(stderr,
                     "  WARNING: no IC source declared (set `ic = <kind>` in the case file)\n");

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

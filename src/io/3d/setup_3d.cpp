#include "io/3d/setup_3d.h"
#include "io/3d/freestream.h"
#include "io/3d/plate.h"
#include "io/3d/trefoil_knot.h"
#include "io/3d/vortex_ring.h"
#include "io/vtk_slim_3d.h"
#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace scene3d {

// ── Author cross-check: load a shared staggered IC (ic{x,y,z}.raw, float32) ──
// into the grid's MAC faces — the SAME layout write_face_ic dumps.
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
// Each case file declares one or more IC SOURCES; every source picks a primitive
// by `kind` and reads its params. Decouples "what the IC is" (data) from "how to
// build it" (the analytic math in io/3d/*, which we only call).
//
// Case-file syntax (index N = 0,1,2,…):
//   ic    = vortex_ring     # kind of source 0 (alias for ic0)
//   ic1   = vortex_ring     # kind of source 1
//   ic0.center = 0.4,0.5,0.5
// A source N's params are read from `icN.<param>`; for source 0 the unindexed
// `ic.<param>` is accepted too. Missing keys fall back to the builder's default.
namespace {
struct IcParams {
    const Config* cfg;
    std::string prefix;  // "ic0", "ic1", … (the indexed prefix for this source)
    std::string prefix0; // "ic" for source 0, else "" (the unindexed alias)

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

using IcBuilder = std::function<void(Grid3D&, const Config&, const IcParams&)>;

// kind "vortex_ring": Gaussian-cored ring via Biot-Savart.
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

// kind "plate": immersed flat-plate solid (or an external SDF) + uniform
// freestream IC from inflow_*.
static void build_plate(Grid3D& g, const Config& cfg, const IcParams& p) {
    std::string sdf = p.s("sdf_path", "");
    if (!sdf.empty()) {
        scenarios::load_sdf_solid(g, sdf); // exact geometry from a data file
    } else {
        scenarios::Plate plate;
        plate.leading_x = p.d("leading_x", 0.5);
        plate.chord     = p.d("chord", 1.0);
        plate.semi_span = p.d("semi_span", 0.35);
        plate.thickness = p.d("thickness", 0.02);
        plate.tilt_deg  = p.d("tilt_deg", 20.0);
        plate.y_mid     = p.d("y_mid", 0.5);
        scenarios::setup_plate(g, plate);
    }
    scenarios::set_uniform_freestream(g, cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
}

static const std::unordered_map<std::string, IcBuilder>& ic_registry() {
    static const std::unordered_map<std::string, IcBuilder> kReg = {
        {"vortex_ring", build_vortex_ring},
        {"trefoil_knot", build_trefoil_knot},
        {"plate", build_plate},
    };
    return kReg;
}

void setup(Grid3D& g, const Config& cfg) {
    // If ic_dir is set, load that exact staggered field and skip the analytic IC.
    std::string ic_dir = cfg.sget("ic_dir", "");
    if (!ic_dir.empty()) {
        if (load_raw_ic(g, ic_dir)) {
            std::printf("  loaded shared IC from %s/ (cross-check field)\n", ic_dir.c_str());
            return;
        }
        std::fprintf(stderr, "  WARNING: ic_dir='%s' unreadable — falling back to analytic IC\n",
                     ic_dir.c_str());
    }

    // Walk the IC sources the case declares: source 0 is `ic`, then ic1, ic2, …
    int applied = 0;
    for (int n = 0;; n++) {
        std::string kind_key = (n == 0) ? "ic" : ("ic" + std::to_string(n));
        std::string kind     = cfg.sget(kind_key, "");
        if (kind.empty())
            break;
        auto it = ic_registry().find(kind);
        if (it == ic_registry().end())
            throw std::runtime_error("cfdsim: unknown IC kind '" + kind + "' for " + kind_key);
        IcParams p{&cfg, "ic" + std::to_string(n), (n == 0) ? "ic" : ""};
        it->second(g, cfg, p);
        applied++;
    }
    if (applied == 0)
        std::fprintf(stderr,
                     "  WARNING: no IC source declared (set `ic = <kind>` in the case file)\n");

    // Cross-check export: dump the exact IC the solver will run (MAC faces).
    std::string dump_ic = cfg.sget("dump_ic_dir", "");
    if (!dump_ic.empty()) {
        io3d::write_face_ic(g, dump_ic);
        std::printf("  dumped face IC to %s/ic{x,y,z}.raw\n", dump_ic.c_str());
    }
}

bc::BoundaryManager3D make_cpu_bcs(const Config& cfg) {
    if (cfg.lfm_bc == "freestream")
        return scenarios::freestream_box_bcs(cfg.inflow_ux, cfg.inflow_uy, cfg.inflow_uz);
    return bc::free_slip_box();
}

} // namespace scene3d

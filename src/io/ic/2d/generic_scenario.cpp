#include "io/ic/2d/generic_scenario.h"
#include "io/ic/2d/params_2d.h"
#include "io/ic/2d/primitives_2d.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace scenarios {
namespace {

// Collect an index-style primitive list under `prefix`:
//   prefix      = <kind>   (the 0th source's kind; single-source shorthand)
//   prefix0..N  = <kind>   (explicit per-index kinds; prefix0 overrides prefix)
// Returns {index, kind} pairs in ascending index order, scanning until the
// first missing index. Empty if no `prefix`/`prefix0` key is present.
std::vector<std::pair<int, std::string>> collect_list(const Config& cfg, const std::string& prefix) {
    std::vector<std::pair<int, std::string>> out;
    for (int idx = 0;; ++idx) {
        std::string indexed = prefix + std::to_string(idx);
        std::string kind    = cfg.sget(indexed, "");
        if (kind.empty() && idx == 0)
            kind = cfg.sget(prefix, ""); // single-source shorthand
        if (kind.empty())
            break;
        out.emplace_back(idx, kind);
    }
    return out;
}

template <class Reg>
const typename Reg::mapped_type& lookup(const Reg& reg, const std::string& kind,
                                        const char* what) {
    auto it = reg.find(kind);
    if (it == reg.end())
        throw std::runtime_error(std::string("config: unknown 2D ") + what + " primitive '" + kind +
                                 "'");
    return it->second;
}

} // namespace

void GenericScenario::configure(Config& cfg) const {
    // Domain/time presets are data. NY and dt are commonly derived from NX, so
    // support the two historical rules via knobs; anything else the case file
    // sets directly (Lx, Ly, dt, time_integrator, solver, out_dir, …).
    int ny_ratio = cfg.iget("ny_ratio", 0);
    int ny_min   = cfg.iget("ny_min", 16);
    if (ny_ratio > 0)
        cfg.NY = std::max(cfg.NX / ny_ratio, ny_min);
    else if (cfg.iget("ny_square", 0) != 0)
        cfg.NY = cfg.NX;

    double dt_cfl = cfg.dget("dt_cfl", 0.0);
    if (dt_cfl > 0.0)
        cfg.dt = dt_cfl * (cfg.Lx / cfg.NX) / cfg.U_inf;

    cfg.out_dir = cfg.sget("out_dir", cfg.out_dir);
}

void GenericScenario::init_grid(Grid& g, const Config& cfg) const {
    // Geometry first (marks solids), then the additive velocity IC.
    for (const auto& [idx, kind] : collect_list(cfg, "geom"))
        lookup(geom_registry(), kind, "geometry")(g, cfg, Param2D(cfg, "geom", idx));
    for (const auto& [idx, kind] : collect_list(cfg, "ic"))
        lookup(ic_registry(), kind, "IC")(g, cfg, Param2D(cfg, "ic", idx));
}

bc::BoundaryManager GenericScenario::boundary_manager(const Config& cfg) const {
    std::string kind = cfg.sget("bc", "free_slip_walls");
    return lookup(bc_registry(), kind, "boundary")(cfg);
}

void GenericScenario::apply_body_force(Grid& g, const Config& cfg) const {
    for (const auto& [idx, kind] : collect_list(cfg, "body_force"))
        lookup(force_registry(), kind, "body-force")(g, cfg, Param2D(cfg, "body_force", idx));
}

} // namespace scenarios

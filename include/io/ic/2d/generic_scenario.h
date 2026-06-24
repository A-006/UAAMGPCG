#pragma once
#include "io/ic/scenario.h"
#include <string>

namespace scenarios {

// ──────────────────────────────────────────────────────────────────
// GenericScenario — the one and only 2D scenario implementation.
//
// It owns no scene-specific knowledge: domain/time presets, geometry, the
// initial condition, the boundary stack and any body force are all DATA, read
// from cfg / cfg.extra (populated by a case file such as inputs/karman.in).
// Geometry/IC/force "kinds" are looked up in the primitive registries
// (primitives_2d.h). Adding a 2D scene is therefore a new inputs/<name>.in —
// no C++ changes and no per-name branching anywhere.
//
// Key schema (all optional; sensible built-in defaults reproduce the historical
// hardcoded scenarios bit-for-bit):
//   Domain/time presets (configure):
//     ny_ratio   NY = max(NX / ny_ratio, ny_min)   (default ny_ratio = 0 ⇒ NY = NX)
//     ny_min     floor for the derived NY            (default 16)
//     dt_cfl     dt = dt_cfl * (Lx/NX) / U_inf       (default 0 ⇒ leave dt as-is)
//     (Lx, Ly, dt, time_integrator, solver, out_dir use the normal Config keys)
//   Geometry list:  geom / geom0 / geom1 … = <kind>, with geomN.<param> knobs
//   IC list:        ic   / ic0   / ic1   … = <kind>, with icN.<param> knobs
//   Body force:     body_force = <kind>, with body_force.<param> knobs
//   Boundary:       bc = karman | smoke | free_slip_walls
// ──────────────────────────────────────────────────────────────────
class GenericScenario : public Scenario {
public:
    void configure(Config& cfg) const override;
    void init_grid(Grid& g, const Config& cfg) const override;
    bc::BoundaryManager boundary_manager(const Config& cfg) const override;
    void apply_body_force(Grid& g, const Config& cfg) const override;
    const char* name() const override {
        return "generic";
    }
};

} // namespace scenarios

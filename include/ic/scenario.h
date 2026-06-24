#pragma once
#include "core/bc/boundary_condition.h"
#include "core/config.h"
#include "core/grid.h"

// ──────────────────────────────────────────────────────────────────
// Scenario — a first-class object owning everything scenario-specific
// for a 2D run: domain/time presets, grid initialization, the boundary
// stack, and optional per-step body forces.
//
// Adding a new scenario = one new file that subclasses Scenario plus one
// registration line in register_builtin_scenarios(); no edits to the
// simulators or the CLI. See [scenario_registry.h].
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

class Scenario {
public:
    virtual ~Scenario() = default;

    // Apply domain / time-step presets onto cfg (Lx, Ly, NY rule, dt rule,
    // out_dir). Called after the CLI has filled NX / U_inf, so derived
    // fields (NY, dt) may read them. Touch only scenario-owned fields.
    virtual void configure(Config& cfg) const = 0;

    // Initialize the grid: mark solids, set the initial velocity field,
    // seed any perturbation.
    virtual void init_grid(Grid& g, const Config& cfg) const = 0;

    // Build the boundary-condition stack applied every step.
    virtual bc::BoundaryManager boundary_manager(const Config& cfg) const = 0;

    // Optional body force added once per step (e.g. smoke buoyancy).
    virtual void apply_body_force(Grid&, const Config&) const {}

    virtual const char* name() const = 0;
};

} // namespace scenarios

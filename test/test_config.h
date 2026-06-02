#pragma once
#include "config/config.h"
#include "simulator/scenarios/2d/karman_scenario.h"
#include "simulator/scenarios/2d/smoke_scenario.h"

// ──────────────────────────────────────────────────────────────────
// Shared Config builders for tests. They reuse the SAME scenario presets
// (Lx/Ly/NY/dt/out_dir) as the production CLI, so a test never drifts from
// the real configuration. Re/cyl_*/U_inf come from Config defaults; tests
// override individual fields (solver, solve_iters, cyl_R, ...) after.
// ──────────────────────────────────────────────────────────────────

// Cylinder in a 4×1 channel at Re=200 (the canonical Kármán case).
inline Config make_karman_config(int NX = 64, double t_end = 0.1) {
    Config cfg;
    cfg.scenario = "karman";
    cfg.NX       = NX;
    cfg.t_end    = t_end;
    scenarios::KarmanScenario().configure(cfg);
    return cfg;
}

// Buoyant smoke in a closed unit box.
inline Config make_smoke_config(int NX = 64, double t_end = 0.1) {
    Config cfg;
    cfg.scenario = "smoke";
    cfg.NX       = NX;
    cfg.t_end    = t_end;
    scenarios::SmokeScenario().configure(cfg);
    return cfg;
}

#pragma once
#include "io/cli.h"
#include "io/config.h"

// ──────────────────────────────────────────────────────────────────
// Shared Config builders for tests. They go through the SAME production
// config path as the CLI (config::build → scenario case file inputs/<name>.in
// + overrides), so a test never drifts from the real configuration. Callers
// override individual fields (solver, solve_iters, cyl_R, ...) afterwards.
// ──────────────────────────────────────────────────────────────────

// Cylinder in a 4×1 channel at Re=200 (the canonical Kármán case).
inline Config make_karman_config(int NX = 64, double t_end = 0.1) {
    return config::build({{"scenario", "karman"},
                          {"NX", std::to_string(NX)},
                          {"t_end", std::to_string(t_end)}});
}

// Buoyant smoke in a closed unit box.
inline Config make_smoke_config(int NX = 64, double t_end = 0.1) {
    return config::build({{"scenario", "smoke"},
                          {"NX", std::to_string(NX)},
                          {"t_end", std::to_string(t_end)}});
}

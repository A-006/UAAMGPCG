#pragma once
#include "lfm/grid.h"
#include "lfm/config.h"

// Pressure projection: project velocity onto divergence-free subspace.
//   1. rhs = ∇·ũ / Δt
//   2. Solve ∇²p = rhs (using solver specified in cfg)
//   3. u = ũ - Δt·∇p
void pressure_projection(Grid& g, double dt, const Config& cfg);

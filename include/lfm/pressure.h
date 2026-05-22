#pragma once
#include "lfm/grid.h"

// Pressure projection: project velocity onto divergence-free subspace.
//   1. rhs = ∇·ũ / Δt
//   2. Solve ∇²p = rhs (Jacobi)
//   3. u = ũ - Δt·∇p
void pressure_projection(Grid& g, double dt, int jacobi_iters);

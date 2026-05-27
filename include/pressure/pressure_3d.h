#pragma once
#include "core/grid_3d.h"
#include "solver/solver_3d.h"

// 3D pressure projection — parallels include/pressure/pressure.h.
//   1. rhs = ∇·ũ / Δt
//   2. Solve ∇²p = rhs using the provided Solver3D
//   3. u = ũ - Δt·∇p     (and same for v, w)
class PressureProjection3D {
public:
    static void project(Grid3D& g, double dt, Solver3D& solver,
                        int max_iter, double tol);
};

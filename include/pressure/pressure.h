#pragma once
#include "core/grid.h"
#include "solver/solver.h"

// Pressure projection: enforce incompressibility ∇·u = 0.
// Corresponds to OpenFOAM's solution algorithm (PISO/SIMPLE).
//
//   1. rhs = ∇·ũ / Δt
//   2. Solve ∇²p = rhs using the provided solver
//   3. u = ũ - Δt·∇p
class PressureProjection {
public:
    static void project(Grid& g, double dt, Solver& solver,
                        int max_iter, double tol);
};

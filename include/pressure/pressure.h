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

    // Recover the static (kinematic) pressure from a divergence-free velocity
    // field via the pressure Poisson equation (ρ=1):
    //   ∇²p = -[(∂u/∂x)² + 2(∂u/∂y)(∂v/∂x) + (∂v/∂y)²].
    // Needed for force diagnostics in the impulse/LFM method, where the
    // projection's scalar field is the gauge potential, not the static pressure.
    static void recoverStaticPressure(Grid& g, Solver& solver,
                                      int max_iter, double tol);
};

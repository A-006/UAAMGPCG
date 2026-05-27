#pragma once
#include "bc/boundary_condition.h"

// ──────────────────────────────────────────────────────────────────
// Concrete BCs for 2D simulations.
// ──────────────────────────────────────────────────────────────────
namespace bc {

// Fixed-value inflow on the LEFT patch (x=0): u = U_inf, v = 0.
class InflowLeft : public BoundaryCondition {
public:
    explicit InflowLeft(double U_inf) : U_inf_(U_inf) {}
    void apply(Grid& g) const override;
    const char* name() const override { return "InflowLeft"; }
private:
    double U_inf_;
};

// Zero-gradient outflow on the RIGHT patch (x=Lx): ∂u/∂x = 0, ∂v/∂x = 0.
class OutflowRight : public BoundaryCondition {
public:
    void apply(Grid& g) const override;
    const char* name() const override { return "OutflowRight"; }
};

// Free-slip wall on TOP and BOTTOM patches (y=0, y=Ly):
//   v = 0 on the wall, ∂u/∂y = 0 (tangential velocity unchanged).
class FreeSlipTopBottom : public BoundaryCondition {
public:
    void apply(Grid& g) const override;
    const char* name() const override { return "FreeSlipTopBottom"; }
};

// No-slip wall on TOP and BOTTOM (used by Smoke scenario):
//   v = 0, u_ghost = -u_inner (reflect for zero on wall).
class NoSlipTopBottom : public BoundaryCondition {
public:
    void apply(Grid& g) const override;
    const char* name() const override { return "NoSlipTopBottom"; }
};

// No-slip wall on LEFT and RIGHT (used by Smoke scenario):
//   u = 0 on the wall, v_ghost = -v_inner.
class NoSlipLeftRight : public BoundaryCondition {
public:
    void apply(Grid& g) const override;
    const char* name() const override { return "NoSlipLeftRight"; }
};

// No-slip on every fluid-solid face inside the domain (immersed obstacles).
// Sets u/v faces touching a solid cell from a fluid cell to zero.
class NoSlipImmersedSolid : public BoundaryCondition {
public:
    void apply(Grid& g) const override;
    const char* name() const override { return "NoSlipImmersedSolid"; }
};

// ── Scenario builders: return a fully-constructed BoundaryManager ──
BoundaryManager karman(double U_inf);  // inflow + outflow + slip walls + solid
BoundaryManager smoke();               // four no-slip walls + solid

}  // namespace bc

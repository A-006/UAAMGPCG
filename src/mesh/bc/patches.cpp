#include "mesh/bc/patches.h"

namespace bc {

// ── InflowLeft ──────────────────────────────────────────────────
void InflowLeft::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j) = U_inf_;
        g.v_at(0, j) = 0.0;
    }
    // Ghost row v at left corner (j=0 and j=ny) — keep symmetric
    g.v_at(0, 0)  = 0.0;
    g.v_at(0, ny) = 0.0;
    (void)nx;
}

// ── OutflowRight ────────────────────────────────────────────────
void OutflowRight::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int j = 1; j <= ny; j++) {
        g.u_at(nx, j)     = g.u_at(nx - 1, j);
        g.v_at(nx + 1, j) = g.v_at(nx, j);
    }
    g.v_at(nx + 1, 0)  = 0.0;
    g.v_at(nx + 1, ny) = 0.0;
}

// ── FreeSlipTopBottom ───────────────────────────────────────────
void FreeSlipTopBottom::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    // ∂u/∂y = 0 → ghost u row copies the interior row
    for (int i = 0; i <= nx; i++) {
        g.u_at(i, 0)      = g.u_at(i, 1);
        g.u_at(i, ny + 1) = g.u_at(i, ny);
    }
    // v = 0 on wall
    for (int i = 1; i <= nx; i++) {
        g.v_at(i, 0)  = 0.0;
        g.v_at(i, ny) = 0.0;
    }
}

// ── NoSlipTopBottom ─────────────────────────────────────────────
void NoSlipTopBottom::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    // u_ghost = -u_inner (reflection → zero on wall midline)
    for (int i = 0; i <= nx; i++) {
        g.u_at(i, 0)      = -g.u_at(i, 1);
        g.u_at(i, ny + 1) = -g.u_at(i, ny);
    }
    for (int i = 1; i <= nx; i++) {
        g.v_at(i, 0)  = 0.0;
        g.v_at(i, ny) = 0.0;
    }
}

// ── NoSlipLeftRight ─────────────────────────────────────────────
void NoSlipLeftRight::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j)  = 0.0;
        g.u_at(nx, j) = 0.0;
    }
    for (int j = 0; j <= ny; j++) {
        g.v_at(0, j)      = -g.v_at(1, j);
        g.v_at(nx + 1, j) = -g.v_at(nx, j);
    }
}

// ── NoSlipImmersedSolid ─────────────────────────────────────────
// Zero ALL four MAC faces of every solid cell, unconditionally — matching the
// author's SetBcByPhiKernel (each phi<0 cell has its surrounding faces set to 0).
// The old version guarded each face with `!is_solid(neighbor)`, so it only zeroed
// fluid-facing interface faces and left solid↔solid (and inflow-side) faces at the
// freestream value — the 2D analogue of the delta-wing solid-BC gap.
void NoSlipImmersedSolid::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (!g.is_solid(i, j))
                continue;
            g.u_at(i - 1, j) = 0.0; // x- face
            g.u_at(i, j)     = 0.0; // x+ face
            g.v_at(i, j - 1) = 0.0; // y- face
            g.v_at(i, j)     = 0.0; // y+ face
        }
    }
}

void FreeSlipLeftRight::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    // u = 0 on wall (no through-flow)
    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j)  = 0.0;
        g.u_at(nx, j) = 0.0;
    }
    // ∂v/∂x = 0 → ghost v column copies the interior column
    for (int j = 0; j <= ny; j++) {
        g.v_at(0, j)      = g.v_at(1, j);
        g.v_at(nx + 1, j) = g.v_at(nx, j);
    }
}

// ── Scenario builders ───────────────────────────────────────────
BoundaryManager free_slip_walls() {
    BoundaryManager mgr;
    mgr.add(std::make_unique<FreeSlipLeftRight>());
    mgr.add(std::make_unique<FreeSlipTopBottom>());
    mgr.add(std::make_unique<NoSlipImmersedSolid>());
    return mgr;
}

BoundaryManager karman(double U_inf) {
    BoundaryManager mgr;
    mgr.add(std::make_unique<InflowLeft>(U_inf));
    mgr.add(std::make_unique<OutflowRight>());
    mgr.add(std::make_unique<FreeSlipTopBottom>());
    mgr.add(std::make_unique<NoSlipImmersedSolid>());
    return mgr;
}

BoundaryManager smoke() {
    BoundaryManager mgr;
    mgr.add(std::make_unique<NoSlipLeftRight>());
    mgr.add(std::make_unique<NoSlipTopBottom>());
    mgr.add(std::make_unique<NoSlipImmersedSolid>());
    return mgr;
}

} // namespace bc

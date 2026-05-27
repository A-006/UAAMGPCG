#include "bc/patches.h"

namespace bc {

// ── InflowLeft ──────────────────────────────────────────────────
void InflowLeft::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j) = U_inf_;
        g.v_at(0, j) = 0.0;
    }
    // Ghost row v at left corner (j=0 and j=ny) — keep symmetric
    g.v_at(0, 0) = 0.0;
    g.v_at(0, ny) = 0.0;
    (void)nx;
}

// ── OutflowRight ────────────────────────────────────────────────
void OutflowRight::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int j = 1; j <= ny; j++) {
        g.u_at(nx, j)   = g.u_at(nx - 1, j);
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
void NoSlipImmersedSolid::apply(Grid& g) const {
    int nx = g.nx, ny = g.ny;
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (!g.is_solid(i, j)) continue;
            if (i > 1  && !g.is_solid(i - 1, j))  g.u_at(i - 1, j) = 0.0;
            if (i < nx && !g.is_solid(i + 1, j))  g.u_at(i, j)     = 0.0;
            if (j > 1  && !g.is_solid(i, j - 1))  g.v_at(i, j - 1) = 0.0;
            if (j < ny && !g.is_solid(i, j + 1))  g.v_at(i, j)     = 0.0;
        }
    }
}

// ── Scenario builders ───────────────────────────────────────────
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

}  // namespace bc

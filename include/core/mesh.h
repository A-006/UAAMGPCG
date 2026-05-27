#pragma once
#include <algorithm>

// ──────────────────────────────────────────────────────────────────
// 2D structured MAC mesh (topology only — no field data).
//
// NX × NY cells, uniform spacing. One layer of ghost cells on each side.
// Storage layouts (column-major, i stride = 1):
//   u-face:  (NX+1) × (NY+2)
//   v-face:  (NX+2) × (NY+1)
//   cell:    (NX+2) × (NY+2)
//
// Grid (defined in core/grid.h) inherits from Mesh2D and adds u/v/p data.
// New code that only needs topology / indexing can take `const Mesh2D&`
// instead of `const Grid&`.
// ──────────────────────────────────────────────────────────────────
struct Mesh2D {
    int nx, ny;
    double dx, dy;

    Mesh2D(int nx_, int ny_, double lx, double ly)
        : nx(nx_), ny(ny_), dx(lx / nx_), dy(ly / ny_) {}

    double Lx() const { return nx * dx; }
    double Ly() const { return ny * dy; }

    // Storage sizes
    int u_size() const { return (nx + 1) * (ny + 2); }
    int v_size() const { return (nx + 2) * (ny + 1); }
    int p_size() const { return (nx + 2) * (ny + 2); }

    // Flat indices (column-major)
    int iu(int i, int j) const { return i + j * (nx + 1); }
    int iv(int i, int j) const { return i + j * (nx + 2); }
    int ip(int i, int j) const { return i + j * (nx + 2); }

    // Cell-center physical coordinates
    double cell_x(int i) const { return (i - 0.5) * dx; }
    double cell_y(int j) const { return (j - 0.5) * dy; }

    // Utility clamps used throughout the solver
    static int    clamp(int    v, int    lo, int    hi) { return v < lo ? lo : (v > hi ? hi : v); }
    static double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
};

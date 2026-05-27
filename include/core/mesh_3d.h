#pragma once
#include <algorithm>

// ──────────────────────────────────────────────────────────────────
// 3D structured MAC mesh (topology only — no field data).
//
// NX × NY × NZ cells, uniform spacing. One layer of ghost cells per side.
// MAC layouts (column-major, i stride = 1):
//   u-face: (NX+1) × (NY+2) × (NZ+2)
//   v-face: (NX+2) × (NY+1) × (NZ+2)
//   w-face: (NX+2) × (NY+2) × (NZ+1)
//   cell:   (NX+2) × (NY+2) × (NZ+2)
//
// Grid3D (core/grid_3d.h) inherits from Mesh3D and adds field data.
// ──────────────────────────────────────────────────────────────────
struct Mesh3D {
    int nx, ny, nz;
    double dx, dy, dz;

    Mesh3D(int nx_, int ny_, int nz_, double lx, double ly, double lz)
        : nx(nx_), ny(ny_), nz(nz_),
          dx(lx / nx_), dy(ly / ny_), dz(lz / nz_) {}

    double Lx() const { return nx * dx; }
    double Ly() const { return ny * dy; }
    double Lz() const { return nz * dz; }

    // Storage sizes
    int u_size() const { return (nx + 1) * (ny + 2) * (nz + 2); }
    int v_size() const { return (nx + 2) * (ny + 1) * (nz + 2); }
    int w_size() const { return (nx + 2) * (ny + 2) * (nz + 1); }
    int p_size() const { return (nx + 2) * (ny + 2) * (nz + 2); }

    // Flat indices (column-major)
    int iu(int i, int j, int k) const { return i + j * (nx + 1) + k * (nx + 1) * (ny + 2); }
    int iv(int i, int j, int k) const { return i + j * (nx + 2) + k * (nx + 2) * (ny + 1); }
    int iw(int i, int j, int k) const { return i + j * (nx + 2) + k * (nx + 2) * (ny + 2); }
    int ip(int i, int j, int k) const { return i + j * (nx + 2) + k * (nx + 2) * (ny + 2); }

    // Cell-center physical coordinates
    double cell_x(int i) const { return (i - 0.5) * dx; }
    double cell_y(int j) const { return (j - 0.5) * dy; }
    double cell_z(int k) const { return (k - 0.5) * dz; }

    static int    clamp(int    v, int    lo, int    hi) { return v < lo ? lo : (v > hi ? hi : v); }
    static double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
};

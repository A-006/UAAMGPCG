#pragma once
#include "core/mesh_3d.h"
#include <vector>
#include <cstddef>

// ──────────────────────────────────────────────────────────────────
// 3D MAC grid — legacy facade.
//
// Grid3D : public Mesh3D adds u/v/w/p data plus solid mask. Layout mirrors
// 2D Grid : public Mesh2D — new code that only needs topology should take
// `const Mesh3D&`. Pressure solvers and preconditioners under
// src/solver/*_3d.cpp consume Grid3D directly.
// ──────────────────────────────────────────────────────────────────
struct Grid3D : public Mesh3D {
    std::vector<double> u, v, w, p;
    std::vector<bool>   solid;

    Grid3D(int nx_, int ny_, int nz_, double lx, double ly, double lz);

    double  u_at(int i, int j, int k) const { return u[iu(i, j, k)]; }
    double& u_at(int i, int j, int k)       { return u[iu(i, j, k)]; }
    double  v_at(int i, int j, int k) const { return v[iv(i, j, k)]; }
    double& v_at(int i, int j, int k)       { return v[iv(i, j, k)]; }
    double  w_at(int i, int j, int k) const { return w[iw(i, j, k)]; }
    double& w_at(int i, int j, int k)       { return w[iw(i, j, k)]; }
    double  p_at(int i, int j, int k) const { return p[ip(i, j, k)]; }
    double& p_at(int i, int j, int k)       { return p[ip(i, j, k)]; }

    bool is_solid(int i, int j, int k) const { return solid[ip(i, j, k)]; }
    void set_solid(int i, int j, int k)       { solid[ip(i, j, k)] = true; }

    // Divergence — kept for back-compat; new code uses fvc::divergence (3D variant).
    double divergence(int i, int j, int k) const {
        return (u_at(i, j, k) - u_at(i - 1, j, k)) / dx
             + (v_at(i, j, k) - v_at(i, j - 1, k)) / dy
             + (w_at(i, j, k) - w_at(i, j, k - 1)) / dz;
    }
};

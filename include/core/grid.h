#pragma once
#include "core/mesh.h"
#include <vector>

// ──────────────────────────────────────────────────────────────────
// 2D MAC grid — legacy facade.
//
// Grid IS A Mesh2D (inherits nx/ny/dx/dy + index helpers) plus the field
// data (u, v, p, solid). New code that only needs topology should take
// `const Mesh2D&`; new code that needs typed fields should use
// `fields::FaceXField` / `fields::CellField`.
//
// Grid's data members remain `std::vector<double>` so the legacy
// `grid.u[ip]` access pattern keeps working throughout the codebase.
// ──────────────────────────────────────────────────────────────────
class Grid : public Mesh2D {
public:
    std::vector<double> u, v, p;
    std::vector<bool>   solid;

    // Variable Laplacian coefficients for ∇·((1-φ)∇p).
    // When empty, PCG uses uniform coefficients.
    std::vector<double> lap_diag, lap_off_x, lap_off_y;

    Grid(int nx_, int ny_, double lx, double ly);

    const std::vector<double>& u_data() const { return u; }
    const std::vector<double>& v_data() const { return v; }

    bool has_variable_lap() const { return !lap_diag.empty(); }
    void init_variable_lap();

    double  u_at(int i, int j) const { return u[iu(i, j)]; }
    double& u_at(int i, int j)       { return u[iu(i, j)]; }
    double  v_at(int i, int j) const { return v[iv(i, j)]; }
    double& v_at(int i, int j)       { return v[iv(i, j)]; }
    double  p_at(int i, int j) const { return p[ip(i, j)]; }
    double& p_at(int i, int j)       { return p[ip(i, j)]; }

    bool is_solid(int i, int j) const { return solid[ip(i, j)]; }
    void set_solid(int i, int j)       { solid[ip(i, j)] = true; }

    // Divergence — kept for back-compat; new code should use fvc::divergence.
    double divergence(int i, int j) const {
        return (u_at(i, j) - u_at(i - 1, j)) / dx
             + (v_at(i, j) - v_at(i, j - 1)) / dy;
    }
};

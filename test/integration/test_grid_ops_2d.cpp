// Unit tests for the 2D core MAC grid + field operators.
//
// Covers:
//   - Grid / Mesh2D construction (dx, dy, storage sizes)
//   - MAC flat-index helpers (iu / iv / ip) and their stride layout
//   - u_at / v_at / p_at read & write round-trips
//   - solid mask set_solid / is_solid
//   - fvc::divergence on analytic fields (u = x  =>  div = constant)
//   - fvc::vorticity on solid-body rotation (=> constant 2*omega)
//   - fvc::laplacian on a quadratic field
//   - fvc::kinetic_energy on a uniform field
//   - fields::Field<> typed wrapper (sizes, layout, round-trip)
#include "../test_utils.h"

#include "core/grid.h"
#include "core/mesh.h"
#include "core/field.h"
#include "numerics/ops/operators.h"

int main() {
    test_header("2D core grid + field operators");

    // ── Construction: dx / dy / Lx / Ly ────────────────────────────
    {
        Grid g(4, 8, 2.0, 4.0);
        check_approx(g.dx, 0.5, 1e-12, "dx = lx/nx");
        check_approx(g.dy, 0.5, 1e-12, "dy = ly/ny");
        check(g.nx == 4 && g.ny == 8, "nx/ny stored");
        check_approx(g.Lx(), 2.0, 1e-12, "Lx() = nx*dx");
        check_approx(g.Ly(), 4.0, 1e-12, "Ly() = ny*dy");
    }

    // ── Storage sizes match the documented MAC layout ──────────────
    {
        Grid g(4, 8, 1.0, 1.0);
        check(g.u_size() == (4 + 1) * (8 + 2), "u_size = (nx+1)*(ny+2)");
        check(g.v_size() == (4 + 2) * (8 + 1), "v_size = (nx+2)*(ny+1)");
        check(g.p_size() == (4 + 2) * (8 + 2), "p_size = (nx+2)*(ny+2)");
        check(g.u.size() == (size_t)g.u_size(), "u vector allocated to u_size");
        check(g.v.size() == (size_t)g.v_size(), "v vector allocated to v_size");
        check(g.p.size() == (size_t)g.p_size(), "p vector allocated to p_size");
        check(g.solid.size() == (size_t)g.p_size(), "solid vector allocated to p_size");
    }

    // ── Fresh grid is zero-initialised and solid-free ──────────────
    {
        Grid g(3, 3, 1.0, 1.0);
        bool all_zero = true;
        for (double x : g.u) all_zero &= (x == 0.0);
        for (double x : g.v) all_zero &= (x == 0.0);
        for (double x : g.p) all_zero &= (x == 0.0);
        check(all_zero, "u/v/p initialised to zero");
        bool any_solid = false;
        for (bool s : g.solid) any_solid |= s;
        check(!any_solid, "no solid cells on fresh grid");
    }

    // ── MAC flat-index helpers (column-major, i stride = 1) ────────
    {
        Grid g(4, 8, 1.0, 1.0);
        check(g.iu(0, 0) == 0, "iu(0,0) = 0");
        check(g.iu(1, 0) - g.iu(0, 0) == 1, "iu i-stride = 1");
        check(g.iu(0, 1) - g.iu(0, 0) == (g.nx + 1), "iu j-stride = nx+1");
        check(g.iv(1, 0) - g.iv(0, 0) == 1, "iv i-stride = 1");
        check(g.iv(0, 1) - g.iv(0, 0) == (g.nx + 2), "iv j-stride = nx+2");
        check(g.ip(1, 0) - g.ip(0, 0) == 1, "ip i-stride = 1");
        check(g.ip(0, 1) - g.ip(0, 0) == (g.nx + 2), "ip j-stride = nx+2");
        // Indices stay within their respective storage bounds.
        check(g.iu(g.nx, g.ny + 1) == g.u_size() - 1, "iu max index = u_size-1");
        check(g.iv(g.nx + 1, g.ny) == g.v_size() - 1, "iv max index = v_size-1");
        check(g.ip(g.nx + 1, g.ny + 1) == g.p_size() - 1, "ip max index = p_size-1");
    }

    // ── u_at / v_at / p_at read & write round-trips ────────────────
    {
        Grid g(5, 5, 1.0, 1.0);
        g.u_at(2, 3) = 1.25;
        g.v_at(1, 4) = -2.5;
        g.p_at(3, 2) = 7.0;
        check_approx(g.u_at(2, 3), 1.25, 1e-12, "u_at write/read round-trip");
        check_approx(g.v_at(1, 4), -2.5, 1e-12, "v_at write/read round-trip");
        check_approx(g.p_at(3, 2), 7.0, 1e-12, "p_at write/read round-trip");
        // Accessors must alias the underlying flat vectors.
        check_approx(g.u[g.iu(2, 3)], 1.25, 1e-12, "u_at aliases u[iu]");
        check_approx(g.p[g.ip(3, 2)], 7.0, 1e-12, "p_at aliases p[ip]");
    }

    // ── solid mask set/get ─────────────────────────────────────────
    {
        Grid g(6, 6, 1.0, 1.0);
        check(!g.is_solid(2, 2), "cell initially fluid");
        g.set_solid(2, 2);
        check(g.is_solid(2, 2), "set_solid marks the cell");
        check(!g.is_solid(3, 2), "neighbour stays fluid");
        check(g.solid[g.ip(2, 2)], "is_solid aliases solid[ip]");
    }

    // ── divergence of u = x  (=> ∂u/∂x = 1, ∂v/∂y = 0) ─────────────
    // u-face at x = i*dx, so u_at(i,j) = i*dx makes du/dx = (i*dx-(i-1)*dx)/dx = 1.
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx; ++i)
                g.u_at(i, j) = i * g.dx;
        // v stays zero.
        for (int j = 1; j <= g.ny; ++j)
            for (int i = 1; i <= g.nx; ++i) {
                check_approx(fvc::divergence(g, i, j), 1.0, 1e-12,
                             "div(u=x) = 1 at interior cell");
                // Grid::divergence member must agree with fvc::divergence.
                check_approx(g.divergence(i, j), fvc::divergence(g, i, j), 1e-12,
                             "Grid::divergence == fvc::divergence");
            }
    }

    // ── divergence of a divergence-free shear field (u=y, v=0) = 0 ──
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx; ++i)
                g.u_at(i, j) = (j - 0.5) * g.dy; // varies in y only
        check_approx(fvc::divergence(g, 4, 4), 0.0, 1e-12,
                     "div(u=y, v=0) = 0 (no x-variation)");
    }

    // ── vorticity of solid-body rotation: u = -omega*y, v = omega*x ─
    // 2D vorticity ω = ∂v/∂x - ∂u/∂y = omega - (-omega) = 2*omega.
    {
        const double omega = 0.5;
        Grid g(16, 16, 4.0, 4.0);
        // u-face at (i*dx, (j-0.5)*dy):  u = -omega * y
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx; ++i) {
                double y = (j - 0.5) * g.dy;
                g.u_at(i, j) = -omega * y;
            }
        // v-face at ((i-0.5)*dx, j*dy):  v = +omega * x
        for (int j = 0; j <= g.ny; ++j)
            for (int i = 0; i <= g.nx + 1; ++i) {
                double x = (i - 0.5) * g.dx;
                g.v_at(i, j) = omega * x;
            }
        // Interior cells away from the clamped border.
        check_approx(fvc::vorticity(g, 8, 8), 2.0 * omega, 1e-12,
                     "vorticity of solid-body rotation = 2*omega");
        check_approx(fvc::vorticity(g, 5, 11), 2.0 * omega, 1e-12,
                     "vorticity constant across interior");
    }

    // ── vorticity of uniform flow = 0 ──────────────────────────────
    {
        Grid g(8, 8, 1.0, 1.0);
        for (double& x : g.u) x = 3.0;
        for (double& x : g.v) x = -1.0;
        check_approx(fvc::vorticity(g, 4, 4), 0.0, 1e-12, "vorticity of uniform flow = 0");
    }

    // ── laplacian of p = x^2 + y^2  => ∇²p = 4 ─────────────────────
    {
        Grid g(16, 16, 4.0, 4.0);
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx + 1; ++i) {
                double x = (i - 0.5) * g.dx;
                double y = (j - 0.5) * g.dy;
                g.p_at(i, j) = x * x + y * y;
            }
        check_approx(fvc::laplacian(g, g.p, 8, 8), 4.0, 1e-9,
                     "laplacian(x^2+y^2) = 4");
        check_approx(fvc::laplacian(g, g.p, 5, 12), 4.0, 1e-9,
                     "laplacian constant across interior");
    }

    // ── laplacian of a linear field = 0 ────────────────────────────
    {
        Grid g(16, 16, 4.0, 4.0);
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx + 1; ++i)
                g.p_at(i, j) = 2.0 * (i - 0.5) * g.dx - 3.0 * (j - 0.5) * g.dy;
        check_approx(fvc::laplacian(g, g.p, 8, 8), 0.0, 1e-9, "laplacian(linear) = 0");
    }

    // ── kinetic_energy of uniform flow ─────────────────────────────
    // uc = 2, vc = 0  => KE = 0.5*(4 + 0) = 2.
    {
        Grid g(8, 8, 1.0, 1.0);
        for (double& x : g.u) x = 2.0;
        // v stays zero.
        check_approx(fvc::kinetic_energy(g, 4, 4), 2.0, 1e-12,
                     "kinetic_energy(u=2,v=0) = 2");
    }

    // ── init_variable_lap allocates coefficient arrays ─────────────
    {
        Grid g(4, 4, 1.0, 1.0);
        check(!g.has_variable_lap(), "no variable Laplacian by default");
        g.init_variable_lap();
        check(g.has_variable_lap(), "has_variable_lap after init");
        check(g.lap_diag.size() == (size_t)g.p_size(), "lap_diag sized to p_size");
        check(g.lap_off_x.size() == (size_t)g.p_size(), "lap_off_x sized to p_size");
        check(g.lap_off_y.size() == (size_t)g.p_size(), "lap_off_y sized to p_size");
    }

    // ── Mesh2D cell-center coordinates ─────────────────────────────
    {
        Mesh2D m(10, 10, 5.0, 5.0); // dx = dy = 0.5
        check_approx(m.cell_x(1), 0.25, 1e-12, "cell_x(1) = 0.5*dx");
        check_approx(m.cell_y(1), 0.25, 1e-12, "cell_y(1) = 0.5*dy");
        check_approx(m.cell_x(2) - m.cell_x(1), m.dx, 1e-12, "cell_x spacing = dx");
    }

    // ── Mesh2D static clamp helpers ────────────────────────────────
    {
        check(Mesh2D::clamp(5, 0, 3) == 3, "int clamp above hi");
        check(Mesh2D::clamp(-2, 0, 3) == 0, "int clamp below lo");
        check(Mesh2D::clamp(2, 0, 3) == 2, "int clamp in range");
        check_approx(Mesh2D::clamp(1.5, 0.0, 1.0), 1.0, 1e-12, "double clamp above hi");
    }

    // ── fields::Field<> typed wrapper: sizes match MAC layout ──────
    {
        Mesh2D m(4, 8, 1.0, 1.0);
        fields::CellField cell(m);
        fields::FaceXField fx(m);
        fields::FaceYField fy(m);
        check(cell.size() == (size_t)m.p_size(), "CellField size = p_size");
        check(fx.size() == (size_t)m.u_size(), "FaceXField size = u_size");
        check(fy.size() == (size_t)m.v_size(), "FaceYField size = v_size");
    }

    // ── fields::Field<> element access round-trip + layout aliasing ─
    {
        Mesh2D m(5, 5, 1.0, 1.0);
        fields::CellField cell(m);
        cell.fill(0.0);
        cell(3, 2) = 9.0;
        check_approx(cell(3, 2), 9.0, 1e-12, "CellField operator() round-trip");
        check_approx(cell.data()[m.ip(3, 2)], 9.0, 1e-12, "CellField aliases data[ip]");

        fields::FaceXField fx(m);
        fx(2, 1) = -4.0;
        check_approx(fx.data()[m.iu(2, 1)], -4.0, 1e-12, "FaceXField aliases data[iu]");

        cell.fill(1.5);
        check_approx(cell(0, 0), 1.5, 1e-12, "fill() sets every element");
        check_approx(cell(m.nx + 1, m.ny + 1), 1.5, 1e-12, "fill() covers ghost cells");
    }

    return test_summary();
}

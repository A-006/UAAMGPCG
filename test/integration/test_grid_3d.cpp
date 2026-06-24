#include "../test_utils.h"
#include "mesh/grid_3d.h"

// Unit tests for Grid3D (3D MAC grid topology + field data, CPU).

int main() {
    test_header("Grid3D — MAC topology, indexing, solids, divergence");

    // Domain 4x5x6 cells over physical box 8 x 10 x 18 -> dx=2, dy=2, dz=3.
    int nx = 4, ny = 5, nz = 6;
    double lx = 8.0, ly = 10.0, lz = 18.0;
    Grid3D g(nx, ny, nz, lx, ly, lz);

    // ── Spacings ────────────────────────────────────────────────
    check_approx(g.dx, 2.0, 1e-12, "dx = lx/nx");
    check_approx(g.dy, 2.0, 1e-12, "dy = ly/ny");
    check_approx(g.dz, 3.0, 1e-12, "dz = lz/nz");
    check_approx(g.Lx(), 8.0, 1e-12, "Lx = nx*dx");
    check_approx(g.Ly(), 10.0, 1e-12, "Ly = ny*dy");
    check_approx(g.Lz(), 18.0, 1e-12, "Lz = nz*dz");

    // ── Storage sizes (with one ghost layer per side) ───────────
    check(g.u_size() == (nx + 1) * (ny + 2) * (nz + 2), "u_size = (nx+1)(ny+2)(nz+2)");
    check(g.v_size() == (nx + 2) * (ny + 1) * (nz + 2), "v_size = (nx+2)(ny+1)(nz+2)");
    check(g.w_size() == (nx + 2) * (ny + 2) * (nz + 1), "w_size = (nx+2)(ny+2)(nz+1)");
    check(g.p_size() == (nx + 2) * (ny + 2) * (nz + 2), "p_size = (nx+2)(ny+2)(nz+2)");

    // The actual data vectors must match the reported sizes.
    check((int)g.u.size() == g.u_size(), "u vector sized to u_size()");
    check((int)g.v.size() == g.v_size(), "v vector sized to v_size()");
    check((int)g.w.size() == g.w_size(), "w vector sized to w_size()");
    check((int)g.p.size() == g.p_size(), "p vector sized to p_size()");
    check((int)g.solid.size() == g.p_size(), "solid mask sized to p_size()");

    // ── Freshly-constructed grid is all zeros / non-solid ───────
    {
        bool all_zero = true;
        for (double x : g.u) all_zero &= (x == 0.0);
        for (double x : g.v) all_zero &= (x == 0.0);
        for (double x : g.w) all_zero &= (x == 0.0);
        for (double x : g.p) all_zero &= (x == 0.0);
        check(all_zero, "all field data initialized to 0");
        bool none_solid = true;
        for (int k = 0; k <= nz + 1; k++)
            for (int j = 0; j <= ny + 1; j++)
                for (int i = 0; i <= nx + 1; i++)
                    none_solid &= !g.is_solid(i, j, k);
        check(none_solid, "no cells solid on construction");
    }

    // ── MAC flat-index helpers (column-major, i stride 1) ───────
    check(g.iu(0, 0, 0) == 0, "iu(0,0,0) = 0");
    check(g.iu(1, 0, 0) == 1, "iu i-stride is 1");
    check(g.iu(0, 1, 0) == (nx + 1), "iu j-stride is (nx+1)");
    check(g.iu(0, 0, 1) == (nx + 1) * (ny + 2), "iu k-stride is (nx+1)(ny+2)");
    check(g.iv(0, 1, 0) == (nx + 2), "iv j-stride is (nx+2)");
    check(g.iw(0, 0, 1) == (nx + 2) * (ny + 2), "iw k-stride is (nx+2)(ny+2)");
    check(g.ip(0, 1, 0) == (nx + 2), "ip j-stride is (nx+2)");
    check(g.ip(0, 0, 1) == (nx + 2) * (ny + 2), "ip k-stride is (nx+2)(ny+2)");

    // ── u_at/v_at/w_at/p_at: read/write round-trips via accessors ─
    g.u_at(2, 3, 4) = 1.25;
    check_approx(g.u_at(2, 3, 4), 1.25, 1e-12, "u_at write/read round-trip");
    check_approx(g.u[g.iu(2, 3, 4)], 1.25, 1e-12, "u_at writes the iu()-indexed slot");

    g.v_at(1, 2, 3) = -2.5;
    check_approx(g.v_at(1, 2, 3), -2.5, 1e-12, "v_at write/read round-trip");
    check_approx(g.v[g.iv(1, 2, 3)], -2.5, 1e-12, "v_at writes the iv()-indexed slot");

    g.w_at(3, 1, 2) = 7.0;
    check_approx(g.w_at(3, 1, 2), 7.0, 1e-12, "w_at write/read round-trip");
    check_approx(g.w[g.iw(3, 1, 2)], 7.0, 1e-12, "w_at writes the iw()-indexed slot");

    g.p_at(2, 2, 2) = 3.5;
    check_approx(g.p_at(2, 2, 2), 3.5, 1e-12, "p_at write/read round-trip");
    check_approx(g.p[g.ip(2, 2, 2)], 3.5, 1e-12, "p_at writes the ip()-indexed slot");

    // ── set_solid / is_solid ────────────────────────────────────
    check(!g.is_solid(2, 2, 2), "cell (2,2,2) not solid before set");
    g.set_solid(2, 2, 2);
    check(g.is_solid(2, 2, 2), "is_solid true after set_solid");
    check(!g.is_solid(1, 2, 2), "neighbor cell remains non-solid");
    check(g.solid[g.ip(2, 2, 2)], "set_solid sets the ip()-indexed mask slot");

    // ── divergence at a cell on a known field ───────────────────
    // div = (u(i)-u(i-1))/dx + (v(j)-v(j-1))/dy + (w(k)-w(k-1))/dz.
    {
        Grid3D h(nx, ny, nz, lx, ly, lz); // dx=2, dy=2, dz=3
        int ci = 2, cj = 2, ck = 2;
        // Set the six bracketing faces of cell (ci,cj,ck) to known values.
        h.u_at(ci, cj, ck)     = 6.0; // u(i)
        h.u_at(ci - 1, cj, ck) = 2.0; // u(i-1)  -> (6-2)/2 = 2
        h.v_at(ci, cj, ck)     = 5.0; // v(j)
        h.v_at(ci, cj - 1, ck) = 1.0; // v(j-1)  -> (5-1)/2 = 2
        h.w_at(ci, cj, ck)     = 9.0; // w(k)
        h.w_at(ci, cj, ck - 1) = 3.0; // w(k-1)  -> (9-3)/3 = 2
        double expected = (6.0 - 2.0) / h.dx + (5.0 - 1.0) / h.dy + (9.0 - 3.0) / h.dz;
        check_approx(h.divergence(ci, cj, ck), expected, 1e-12,
                     "divergence matches MAC finite-difference formula");
        check_approx(h.divergence(ci, cj, ck), 6.0, 1e-12, "divergence numeric value = 2+2+2");

        // Uniform velocity field -> zero divergence everywhere.
        Grid3D uni(nx, ny, nz, lx, ly, lz);
        for (auto& x : uni.u) x = 1.7;
        for (auto& x : uni.v) x = -0.3;
        for (auto& x : uni.w) x = 4.2;
        check_approx(uni.divergence(ci, cj, ck), 0.0, 1e-12, "uniform velocity -> zero divergence");
    }

    // ── cell-center coordinates ─────────────────────────────────
    check_approx(g.cell_x(1), 0.5 * g.dx, 1e-12, "cell_x(1) = 0.5*dx (first interior center)");
    check_approx(g.cell_y(1), 0.5 * g.dy, 1e-12, "cell_y(1) = 0.5*dy");
    check_approx(g.cell_z(1), 0.5 * g.dz, 1e-12, "cell_z(1) = 0.5*dz");

    return test_summary();
}

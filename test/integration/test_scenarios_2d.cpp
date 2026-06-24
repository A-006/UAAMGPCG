// Unit tests for 2D scenario setup helpers (pure field/mask seeding).
//   - scenarios::setup_cylinder / set_uniform_inflow (karman.h)
//   - scenarios::seed_vortex_dipoles (leapfrog.h)
//
// These exercise the ANALYTIC correctness of the CPU setup code only; no
// GPU simulator is constructed.
#include "../test_utils.h"
#include "io/2d/cylinder.h"
#include "io/2d/leapfrog.h"
#include "io/config.h"
#include "mesh/grid_2d.h"
#include <cmath>

static void test_karman_cylinder() {
    // Domain 4x2, fine enough that pi R^2 / (dx dy) is a meaningful count.
    const int nx = 200, ny = 100;
    const double Lx = 4.0, Ly = 2.0;
    Grid g(nx, ny, Lx, Ly);

    scenarios::Cylinder k;
    k.cyl_cx = 2.0;
    k.cyl_cy = 1.0;
    k.cyl_R  = 0.2;
    k.U_inf  = 1.0;

    scenarios::setup_cylinder(g, k);

    // The cell whose center is nearest the cylinder center must be solid.
    int ic = (int)std::round(k.cyl_cx / g.dx + 0.5);
    int jc = (int)std::round(k.cyl_cy / g.dy + 0.5);
    check(g.is_solid(ic, jc), "cell at cylinder center is solid");

    // A cell clearly outside the cylinder (far upstream corner) is not solid.
    check(!g.is_solid(1, 1), "far upstream corner cell is fluid");
    // A cell just outside the radius along +x is not solid.
    int i_out = (int)std::round((k.cyl_cx + 2.0 * k.cyl_R) / g.dx + 0.5);
    check(!g.is_solid(i_out, jc), "cell 2R downstream of center is fluid");

    // A cell well inside the radius (half a radius up) is solid.
    int j_in = (int)std::round((k.cyl_cy + 0.5 * k.cyl_R) / g.dy + 0.5);
    check(g.is_solid(ic, j_in), "cell at +0.5R is solid");

    // Count solid cells; should be ~ pi R^2 / (dx dy) within a few %.
    int n_solid = 0;
    for (int j = 1; j <= g.ny; j++)
        for (int i = 1; i <= g.nx; i++)
            if (g.is_solid(i, j))
                n_solid++;
    double expected = M_PI * k.cyl_R * k.cyl_R / (g.dx * g.dy);
    check(n_solid > 0, "some cells were marked solid");
    check_approx((double)n_solid, expected, 0.06 * expected,
                 "solid-cell count ~ pi R^2/(dx dy) within 6%");

    // Every solid cell center must actually lie inside the radius (no false +).
    bool all_inside = true;
    double R2 = k.cyl_R * k.cyl_R;
    for (int j = 1; j <= g.ny && all_inside; j++)
        for (int i = 1; i <= g.nx && all_inside; i++)
            if (g.is_solid(i, j)) {
                double rx = (i - 0.5) * g.dx - k.cyl_cx;
                double ry = (j - 0.5) * g.dy - k.cyl_cy;
                if (rx * rx + ry * ry >= R2)
                    all_inside = false;
            }
    check(all_inside, "every solid cell center is strictly inside radius");

    // Symmetry: solid mask is symmetric about the cylinder center row.
    // Center is at a cell boundary (cy=1.0, dy=0.02 -> j boundary), so mirror
    // j about center. Use a symmetric mapping jm = 2*jc-1 - j for the row pair.
    bool symmetric = true;
    for (int j = 1; j <= g.ny && symmetric; j++) {
        int jm = (jc - 1) + (jc - j); // mirror across boundary between jc-1 and jc
        if (jm < 1 || jm > g.ny)
            continue;
        for (int i = 1; i <= g.nx && symmetric; i++)
            if (g.is_solid(i, j) != g.is_solid(i, jm))
                symmetric = false;
    }
    check(symmetric, "solid mask is up/down symmetric about cylinder center");
}

static void test_uniform_inflow_2d() {
    Grid g(64, 32, 4.0, 2.0);
    const double U = 1.3;
    scenarios::set_uniform_inflow(g, U);

    // All interior u-faces equal U_inf.
    bool all_u = true;
    for (int j = 1; j <= g.ny && all_u; j++)
        for (int i = 0; i <= g.nx && all_u; i++)
            if (g.u_at(i, j) != U)
                all_u = false;
    check(all_u, "all u-faces equal U_inf after set_uniform_inflow");

    // Spot checks at corners/center of the u-face range.
    check_approx(g.u_at(0, 1), U, 1e-12, "u at (0,1) == U_inf");
    check_approx(g.u_at(g.nx, g.ny), U, 1e-12, "u at (nx,ny) == U_inf");
    check_approx(g.u_at(g.nx / 2, g.ny / 2), U, 1e-12, "u at center == U_inf");

    // v-faces untouched (left at zero by the constructor).
    bool v_zero = true;
    for (int j = 0; j <= g.ny && v_zero; j++)
        for (int i = 1; i <= g.nx && v_zero; i++)
            if (g.v_at(i, j) != 0.0)
                v_zero = false;
    check(v_zero, "v-faces remain zero after set_uniform_inflow");
}

// Max |divergence| of the leapfrog IC on an N x (N/2) grid over Lx=4, Ly=2.
static double leapfrog_max_div(int N) {
    Grid g(N, N / 2, 4.0, 2.0);
    scenarios::seed_vortex_dipoles(g, scenarios::VortexDipoles{});
    double max_div = 0.0;
    for (int j = 1; j <= g.ny; j++)
        for (int i = 1; i <= g.nx; i++)
            max_div = std::max(max_div, std::abs(g.divergence(i, j)));
    return max_div;
}

static void test_leapfrog_ic() {
    // Match the scenario's own preset domain (Lx=4, Ly=2) so the hardcoded
    // vortex positions (x=0.9, 1.4) land inside the domain.
    const int nx = 128, ny = 64;
    Grid g(nx, ny, 4.0, 2.0);

    scenarios::seed_vortex_dipoles(g, scenarios::VortexDipoles{});

    // The IC is a superposition of Lamb-Oseen vortices, which is analytically
    // divergence-free; on the MAC grid only the discrete truncation error
    // remains. That error must (a) be modest and (b) shrink ~quadratically
    // under refinement — the signature of a genuinely solenoidal field.
    double d64  = leapfrog_max_div(64);
    double d128 = leapfrog_max_div(128);
    double d256 = leapfrog_max_div(256);
    check(d128 < 5e-2, "leapfrog IC max|div| is small (< 5e-2 at N=128)");
    check(d128 < 0.5 * d64, "leapfrog max|div| at least halves from N=64 to 128");
    check(d256 < 0.5 * d128, "leapfrog max|div| at least halves from N=128 to 256");
    // Second order: ratio per halving should be ~4; allow >= 3 for off-grid cores.
    check(d64 / d128 > 3.0 && d128 / d256 > 3.0,
          "leapfrog max|div| converges ~2nd order (ratio > 3 per refinement)");

    // Field must be nonzero somewhere (it actually seeded velocity).
    double max_speed = 0.0;
    for (int j = 1; j <= g.ny; j++)
        for (int i = 0; i <= g.nx; i++)
            max_speed = std::max(max_speed, std::abs(g.u_at(i, j)));
    check(max_speed > 1e-3, "leapfrog IC seeded a nonzero u-field");

    // The configuration is symmetric about the axis y = Ly/2 in the sense that
    // each dipole is a +-G pair straddling the axis. The u-velocity (axial) is
    // even about the axis; v-velocity (transverse) is odd. Check u-evenness at
    // an x column away from cores. Axis is at a cell boundary (Ly/2 = 1.0).
    double Ly = g.Ly();
    int icol = nx / 3;
    bool u_even = true;
    for (int j = 1; j <= g.ny / 2 && u_even; j++) {
        // mirror j across axis y=Ly/2: y_j + y_jm = Ly  ->  jm = ny+1-j
        int jm = g.ny + 1 - j;
        double a = g.u_at(icol, j);
        double b = g.u_at(icol, jm);
        if (std::abs(a - b) > 1e-9 * (1.0 + std::abs(a)))
            u_even = false;
    }
    check(u_even, "leapfrog axial u is symmetric (even) about the propagation axis");
    (void)Ly;
}

int main() {
    test_header("2D scenario setup helpers");
    test_karman_cylinder();
    test_uniform_inflow_2d();
    test_leapfrog_ic();
    return test_summary();
}

// Unit tests for 3D scenario setup helpers (pure field/mask seeding).
//   - scenarios::add_vortex_ring (vortex_ring.h)
//   - scenarios::set_uniform_inflow / set_uniform_freestream (delta_wing.h)
//
// CPU setup correctness only; no GPU simulator is constructed.
#include "../test_utils.h"
#include "io/3d/vortex_ring.h"
#include "io/3d/delta_wing.h"
#include "mesh/grid_3d.h"
#include "core/scalar_field_3d.h"
#include <cmath>

// Returns {normalized max|div| = max|div|*dx/maxU, raw maxU} for a ring on
// an N^3 grid over the unit cube. The Biot-Savart field has steep gradients
// near the filament, so the meaningful solenoidal measure is the
// dimensionless div*dx/U (a relative divergence per cell).
static void ring_div_metrics(int N, double& norm_div, double& max_speed) {
    Grid3D g(N, N, N, 1.0, 1.0, 1.0);
    scenarios::VortexRing vr;
    vr.center      = {{0.5, 0.5, 0.5}};
    vr.axis        = {{0.0, 0.0, 1.0}}; // ring in x-y plane, axis along z
    vr.radius      = 0.2;
    vr.core        = 0.05;
    vr.circulation = 1.0;
    vr.n_segments  = 200;
    scenarios::add_vortex_ring(g, vr);

    double max_div = 0.0;
    for (int k = 2; k <= g.nz - 1; k++)
        for (int j = 2; j <= g.ny - 1; j++)
            for (int i = 2; i <= g.nx - 1; i++)
                max_div = std::max(max_div, std::abs(g.divergence(i, j, k)));
    max_speed = 0.0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++)
                max_speed = std::max(max_speed, std::abs(g.u_at(i, j, k)));
    norm_div = max_div * g.dx / max_speed;
}

static void test_vortex_ring_divergence_free() {
    // Biot-Savart velocity of a closed filament is analytically solenoidal.
    // The dimensionless relative divergence (div*dx/U) must be small and
    // shrink under refinement.
    double n32, n40, n64, s32, s40, s64;
    ring_div_metrics(32, n32, s32);
    ring_div_metrics(40, n40, s40);
    ring_div_metrics(64, n64, s64);

    check(n40 < 5e-2, "vortex-ring relative divergence small (div*dx/U < 5e-2 at N=40)");
    check(n40 < n32, "vortex-ring relative divergence decreases N=32 -> 40");
    check(n64 < n40, "vortex-ring relative divergence decreases N=40 -> 64");

    // Field is actually nonzero (the ring seeded velocity).
    check(s40 > 1e-3, "vortex-ring IC seeded a nonzero velocity field");

    // Re-run a single grid for the accumulation/cancellation check below.
    const int n = 40;
    Grid3D g(n, n, n, 1.0, 1.0, 1.0);
    scenarios::VortexRing vr;
    vr.center      = {{0.5, 0.5, 0.5}};
    vr.axis        = {{0.0, 0.0, 1.0}};
    vr.radius      = 0.2;
    vr.core        = 0.05;
    vr.circulation = 1.0;
    vr.n_segments  = 200;
    scenarios::add_vortex_ring(g, vr);

    // add_vortex_ring ACCUMULATES (+=): applying it again with opposite
    // circulation must cancel the field back toward zero.
    scenarios::VortexRing vr2 = vr;
    vr2.circulation = -1.0;
    scenarios::add_vortex_ring(g, vr2);
    double max_after = 0.0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++)
                max_after = std::max(max_after, std::abs(g.u_at(i, j, k)));
    check(max_after < 1e-9, "opposite-circulation ring cancels the field (accumulates)");
}

static void test_uniform_inflow_3d() {
    Grid3D g(24, 24, 24, 1.0, 1.0, 1.0);
    const double U = 0.7;
    scenarios::set_uniform_inflow(g, U);

    // All interior u-faces equal U_inf; v/w faces untouched (zero).
    bool all_u = true;
    for (int k = 1; k <= g.nz && all_u; k++)
        for (int j = 1; j <= g.ny && all_u; j++)
            for (int i = 0; i <= g.nx && all_u; i++)
                if (g.u_at(i, j, k) != U)
                    all_u = false;
    check(all_u, "3D set_uniform_inflow: all u-faces == U_inf");

    bool vw_zero = true;
    for (int k = 0; k <= g.nz && vw_zero; k++)
        for (int j = 0; j <= g.ny && vw_zero; j++)
            for (int i = 1; i <= g.nx && vw_zero; i++)
                if (g.v_at(i, j, k) != 0.0)
                    vw_zero = false;
    check(vw_zero, "3D set_uniform_inflow: v-faces remain zero");

    check_approx(g.u_at(0, 1, 1), U, 1e-12, "3D u at (0,1,1) == U_inf");
    check_approx(g.u_at(g.nx, g.ny, g.nz), U, 1e-12, "3D u at (nx,ny,nz) == U_inf");
}

static void test_uniform_freestream_3d() {
    Grid3D g(20, 20, 20, 1.0, 1.0, 1.0);
    const double Ux = 1.0, Uy = 0.2, Uz = -0.3;
    scenarios::set_uniform_freestream(g, Ux, Uy, Uz);

    bool ok_u = true, ok_v = true, ok_w = true;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++)
                if (g.u_at(i, j, k) != Ux)
                    ok_u = false;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 0; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                if (g.v_at(i, j, k) != Uy)
                    ok_v = false;
    for (int k = 0; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                if (g.w_at(i, j, k) != Uz)
                    ok_w = false;
    check(ok_u, "set_uniform_freestream: all u-faces == Ux");
    check(ok_v, "set_uniform_freestream: all v-faces == Uy");
    check(ok_w, "set_uniform_freestream: all w-faces == Uz");

    // A uniform field is divergence-free everywhere.
    double max_div = 0.0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                max_div = std::max(max_div, std::abs(g.divergence(i, j, k)));
    check(max_div < 1e-12, "uniform freestream is exactly divergence-free");
}

int main() {
    test_header("3D scenario setup helpers");
    test_vortex_ring_divergence_free();
    test_uniform_inflow_3d();
    test_uniform_freestream_3d();
    return test_summary();
}

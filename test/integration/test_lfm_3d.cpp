/**
 * @file test_lfm_3d.cpp
 * @brief 3D LFM (LFMSimulator3D) unit + integration tests.
 *
 * Mirrors the 2D suite (test/integration/test_lfm.cpp) one axis up:
 *   T1: FlowMap3D initialization & identity
 *   T2: Velocity interpolation on the 3D MAC grid (27-point B-spline)
 *   T3: Viscous force on uniform flow
 *   T4: RK4-March (forward + backward) on uniform flow (12-dim ODE)
 *   T5: Pullback roundtrip (Ψ then Φ) on uniform flow
 *   T6: Full LFM cycle on a vortex ring in a free-slip box
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "simulator/lfm/flow_map_3d.h"
#include "simulator/lfm/lfm_simulator_3d.h"
#include "ic/3d/vortex_ring.h"
#include "solver/factory_3d.h"
#include "../test_utils.h"
#include <cmath>
#include <iostream>

static void fill_uniform(Grid3D& g, double uu, double vv, double ww) {
    std::fill(g.u.begin(), g.u.end(), uu);
    std::fill(g.v.begin(), g.v.end(), vv);
    std::fill(g.w.begin(), g.w.end(), ww);
}

// ═══════════════════════════════════════════════════════════
// T1: FlowMap3D identity
// ═══════════════════════════════════════════════════════════
static void t1_flowmap() {
    test_header("T1: FlowMap3D identity");

    FlowMap3D fm(4, 3, 2, 1.0, 0.5, 0.25);
    fm.set_identity();

    size_t k = fm.idx(1, 1, 1); // center (0.5, 0.25, 0.125)
    check(std::abs(fm.phi_x[k] - 0.5) < 1e-12, "phi_x(1,1,1)=0.5");
    check(std::abs(fm.phi_y[k] - 0.25) < 1e-12, "phi_y(1,1,1)=0.25");
    check(std::abs(fm.phi_z[k] - 0.125) < 1e-12, "phi_z(1,1,1)=0.125");
    check(std::abs(fm.F00[k] - 1.0) < 1e-12 && std::abs(fm.F11[k] - 1.0) < 1e-12 &&
              std::abs(fm.F22[k] - 1.0) < 1e-12,
          "F = diag(1,1,1)");
    check(std::abs(fm.F01[k]) + std::abs(fm.F02[k]) + std::abs(fm.F10[k]) + std::abs(fm.F12[k]) +
                  std::abs(fm.F20[k]) + std::abs(fm.F21[k]) <
              1e-12,
          "F off-diagonals = 0");

    size_t k2 = fm.idx(4, 3, 2); // center (3.5, 1.25, 0.375)
    check(std::abs(fm.phi_x[k2] - 3.5) < 1e-12 && std::abs(fm.phi_y[k2] - 1.25) < 1e-12 &&
              std::abs(fm.phi_z[k2] - 0.375) < 1e-12,
          "phi(4,3,2) = cell center");

    fm.set_backward_identity();
    check(std::abs(fm.psi_x[k] - 0.5) < 1e-12, "psi starts at identity");
    check(std::abs(fm.T00[k] - 1.0) < 1e-12 && std::abs(fm.T11[k] - 1.0) < 1e-12 &&
              std::abs(fm.T22[k] - 1.0) < 1e-12,
          "T starts at identity");
}

// ═══════════════════════════════════════════════════════════
// T2: Velocity interpolation (linear field, exact for B-spline)
// ═══════════════════════════════════════════════════════════
static void t2_velocity_interp() {
    test_header("T2: 3D velocity interpolation on MAC grid");

    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0;
    cfg.Re                   = 0;
    cfg.time_integrator      = "lfm";
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));
    Grid3D& g = sim.mutable_grid();

    // Linear fields: u = x, v = -y, w = 2z (reproduced exactly by quadratic B-spline)
    for (int k = 0; k <= 8; k++)
        for (int j = 0; j <= 8; j++)
            for (int i = 0; i <= 8; i++) {
                if (i <= g.nx && j <= g.ny + 1 && k <= g.nz + 1)
                    g.u_at(i, j, k) = i * g.dx;
                if (i <= g.nx + 1 && j <= g.ny && k <= g.nz + 1)
                    g.v_at(i, j, k) = -(j * g.dy);
                if (i <= g.nx + 1 && j <= g.ny + 1 && k <= g.nz)
                    g.w_at(i, j, k) = 2.0 * (k * g.dz);
            }

    // Interior cell center (4,4,4): x=y=z=(4-0.5)*0.25=0.875
    double cx = (4 - 0.5) * g.dx, cy = (4 - 0.5) * g.dy, cz = (4 - 0.5) * g.dz;
    double vu, vv, vw;
    sim.sample_velocity_public(cx, cy, cz, g.u, g.v, g.w, vu, vv, vw);
    check(std::abs(vu - cx) < 0.02, "u at cell center ~ x");
    check(std::abs(vv + cy) < 0.02, "v at cell center ~ -y");
    check(std::abs(vw - 2 * cz) < 0.02, "w at cell center ~ 2z");
}

// ═══════════════════════════════════════════════════════════
// T3: Viscous force on uniform flow ≈ 0
// ═══════════════════════════════════════════════════════════
static void t3_viscous() {
    test_header("T3: Viscous force on uniform flow");

    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0;
    cfg.Re                   = 200;
    cfg.U_inf                = 1.0;
    cfg.cyl_R                = 0.1;
    cfg.time_integrator      = "lfm";
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));

    Grid3D g(8, 8, 8, 2.0, 2.0, 2.0);
    fill_uniform(g, 1.0, 0.0, 0.0);

    std::vector<double> vu, vv, vw;
    sim.compute_viscous_public(g, vu, vv, vw);
    double max_v = 0;
    for (size_t i = 0; i < vu.size(); i++)
        max_v = std::max({max_v, std::abs(vu[i]), std::abs(vv[i]), std::abs(vw[i])});
    check(max_v < 1e-10, "Uniform flow -> viscous ~ 0");
}

// ═══════════════════════════════════════════════════════════
// T4: RK4-March on uniform flow
// ═══════════════════════════════════════════════════════════
static void t4_rk4_march() {
    test_header("T4: RK4-March forward + backward on uniform flow");

    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 16;
    cfg.Lx = cfg.Ly = cfg.Lz = 4.0;
    cfg.Re                   = 0;
    cfg.dt                   = 0.015625;
    cfg.time_integrator      = "lfm";
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));

    Grid3D& g = sim.mutable_grid();
    fill_uniform(g, 1.0, 0.0, 0.0);
    std::vector<double> uu = g.u, vv = g.v, ww = g.w;

    FlowMap3D& fm = sim.flow_map();
    fm.set_identity();
    sim.rk4_march_forward_public(uu, vv, ww, cfg.dt);

    size_t k          = fm.idx(8, 8, 8);
    double x0         = (8 - 0.5) * g.dx;
    double expected_x = x0 + cfg.dt * 1.0;
    check(std::abs(fm.phi_x[k] - expected_x) < 0.001, "Forward march: phi_x ~ x + dt*u");
    check(std::abs(fm.phi_y[k] - (8 - 0.5) * g.dy) < 0.001, "Forward march: phi_y unchanged");
    check(std::abs(fm.F00[k] - 1.0) < 1e-9 && std::abs(fm.F11[k] - 1.0) < 1e-9 &&
              std::abs(fm.F22[k] - 1.0) < 1e-9,
          "F stays identity (uniform flow)");

    fm.set_backward_identity();
    sim.rk4_march_backward_public(uu, vv, ww, -cfg.dt);
    check(std::abs(fm.psi_x[k] - (x0 - cfg.dt)) < 0.001, "Backward march: psi_x ~ x - dt*u");
    check(std::abs(fm.T00[k] - 1.0) < 1e-9, "T stays identity");
}

// ═══════════════════════════════════════════════════════════
// T5: Pullback roundtrip on uniform flow
// ═══════════════════════════════════════════════════════════
static void t5_pullback_roundtrip() {
    test_header("T5: Pullback roundtrip on uniform flow");

    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 16;
    cfg.Lx = cfg.Ly = cfg.Lz = 4.0;
    cfg.Re                   = 0;
    cfg.cyl_R                = 0;
    cfg.dt                   = 0.015625;
    cfg.time_integrator      = "lfm";
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));

    Grid3D& g = sim.mutable_grid();
    fill_uniform(g, 1.0, 0.0, 0.0);
    std::vector<double> uu = g.u, vv = g.v, ww = g.w;

    FlowMap3D& fm = sim.flow_map();
    fm.set_identity();
    for (int s = 0; s < 4; s++)
        sim.rk4_march_forward_public(uu, vv, ww, cfg.dt);
    fm.set_backward_identity();
    for (int s = 0; s < 4; s++)
        sim.rk4_march_backward_public(uu, vv, ww, -cfg.dt);

    sim.pullback_impulse_public(g);

    const auto& mx = sim.impulse_x();
    const auto& my = sim.impulse_y();
    const auto& mz = sim.impulse_z();
    double ex = 0, eyz = 0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                size_t m = fm.idx(i, j, k);
                ex       = std::max(ex, std::abs(mx[m] - 1.0));
                eyz      = std::max({eyz, std::abs(my[m]), std::abs(mz[m])});
            }
    std::cout << "    max|m_x-1|=" << ex << " max|m_y,m_z|=" << eyz << "\n";
    check(ex < 0.1, "Pullback m_x ~ u0 (=1)");
    check(eyz < 0.01, "Pullback m_y, m_z ~ 0");
}

// ═══════════════════════════════════════════════════════════
// T6/T7: Full LFM cycle on a vortex ring (free-slip box, inviscid).
// Parametrized by pressure solver so we exercise both plain "cg" and the
// (now null-space-safe) "pcg_uaamg" preconditioner on the singular box.
// ═══════════════════════════════════════════════════════════
static void vortex_ring_cycle(const char* solver) {
    test_header(std::string("Full LFM cycle on a vortex ring — solver=") + solver);

    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = 32;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0;
    cfg.Re                   = 0; // inviscid: energy should be near-conserved
    cfg.dt                   = 0.25 * (cfg.Lx / cfg.NX) / 1.0;
    cfg.solve_iters          = 500;
    cfg.solve_tol            = 1e-8;
    cfg.time_integrator      = "lfm";
    cfg.lfm_cycle_steps      = 2;
    LFMSimulator3D sim(cfg, Factory3D::create(solver));

    // Vortex ring centered in the box, axis along z.
    scenarios::VortexRing vr;
    vr.center      = {1.0, 1.0, 1.0};
    vr.axis        = {0.0, 0.0, 1.0};
    vr.radius      = 0.4;
    vr.core        = 0.12;
    vr.circulation = 1.0;
    scenarios::add_vortex_ring(sim.mutable_grid(), vr);
    sim.set_boundary_manager(bc::free_slip_box()); // re-apply BC after injecting IC

    auto kinetic_energy = [](const Grid3D& g) {
        double ke = 0;
        for (int k = 1; k <= g.nz; k++)
            for (int j = 1; j <= g.ny; j++)
                for (int i = 1; i <= g.nx; i++) {
                    double uc = 0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k));
                    double vc = 0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k));
                    double wc = 0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1));
                    ke += 0.5 * (uc * uc + vc * vc + wc * wc);
                }
        return ke * g.dx * g.dy * g.dz;
    };
    auto max_speed = [](const Grid3D& g) {
        double s = 0;
        for (int k = 1; k <= g.nz; k++)
            for (int j = 1; j <= g.ny; j++)
                for (int i = 1; i <= g.nx; i++) {
                    double uc = 0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k));
                    double vc = 0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k));
                    double wc = 0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1));
                    s = std::max(s, std::sqrt(uc * uc + vc * vc + wc * wc));
                }
        return s;
    };

    double ke0 = kinetic_energy(sim.grid());
    double sp0 = max_speed(sim.grid());
    std::cout << "    Initial: KE=" << ke0 << " max_speed=" << sp0 << "\n";

    sim.step(); // one full LFM cycle

    const Grid3D& g = sim.grid();
    double ke1 = kinetic_energy(g), sp1 = max_speed(g);
    double max_div = 0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                if (!g.is_solid(i, j, k))
                    max_div = std::max(max_div, std::abs(g.divergence(i, j, k)));

    std::cout << "    Final:   KE=" << ke1 << " (" << ke1 / ke0 * 100 << "% of initial)"
              << " max_speed=" << sp1 << " max|div|=" << max_div << "\n";

    check(std::isfinite(ke1) && std::isfinite(sp1), "Cycle produced finite field");
    check(max_div < 0.1, "Divergence-free after projection (max|div| < 0.1)");
    check(ke1 / ke0 > 0.5 && ke1 / ke0 < 1.5, "Inviscid: kinetic energy near-conserved");
    check(sp1 < 2.0 * sp0, "No velocity blow-up");
}

int main() {
    t1_flowmap();
    t2_velocity_interp();
    t3_viscous();
    t4_rk4_march();
    t5_pullback_roundtrip();
    vortex_ring_cycle("cg");        // T6: reference plain-CG solver
    vortex_ring_cycle("pcg_uaamg"); // T7: UAAMG preconditioner (null-space-safe)
    return test_summary();
}

/**
 * @file test_bfecc_clamp.cpp
 * @brief CPU validation that the BFECC clamp keeps an INVISCID LFM cycle stable.
 *
 * Head-on vortex-ring collision is the stress case: inviscid + no clamp blows up
 * (the paper's reason for BfeccClamp). We run the same inviscid collision twice —
 * clamp OFF then clamp ON — and report max speed each cycle. Expectation:
 *   - clamp OFF: max-speed grows unbounded / NaN within a few dozen cycles.
 *   - clamp ON : max-speed stays bounded; the field remains finite.
 *
 * Usage: test_bfecc_clamp [NX] [cycles]   defaults: NX=48 cycles=60
 */
#include "config/config.h"
#include "core/grid_3d.h"
#include "simulator/lfm_simulator_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include "solver/factory_3d.h"
#include <cmath>
#include <cstdio>
#include <iostream>

static double max_speed(const Grid3D& g) {
    double s = 0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                double uc = 0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k));
                double vc = 0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k));
                double wc = 0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1));
                double sp = std::sqrt(uc * uc + vc * vc + wc * wc);
                if (sp > s)
                    s = sp;
            }
    return s;
}

static double run(bool clamp, int NX, int cycles, double dt_factor) {
    Config cfg;
    cfg.dim = 3;
    cfg.NX = cfg.NY = cfg.NZ = NX;
    cfg.Lx = cfg.Ly = cfg.Lz = 1.0;
    cfg.U_inf                = 1.0;
    cfg.Re                   = 0; // INVISCID — the whole point
    cfg.dt                   = dt_factor * (cfg.Lx / cfg.NX);
    cfg.solve_iters          = 80;
    cfg.solve_tol            = 1e-7;
    cfg.solver               = "cg";
    cfg.time_integrator      = "lfm";
    cfg.lfm_cycle_steps      = 2;
    cfg.lfm_bfecc_clamp      = clamp;

    LFMSimulator3D sim(cfg, Factory3D::create(cfg.solver));
    scenarios::VortexRing left;
    left.center      = {0.36, 0.5, 0.5};
    left.axis        = {1.0, 0.0, 0.0};
    left.radius      = 0.12;
    left.core        = 0.03;
    left.circulation = +1.0;
    left.n_segments  = 200;
    scenarios::VortexRing right = left;
    right.center                = {0.64, 0.5, 0.5};
    right.circulation           = -1.0;
    scenarios::add_vortex_ring(sim.mutable_grid(), left);
    scenarios::add_vortex_ring(sim.mutable_grid(), right);
    sim.set_boundary_manager(bc::free_slip_box());

    double sp0 = max_speed(sim.grid());
    std::printf("  [clamp %s] cycle  0: max_speed=%.4f\n", clamp ? "ON " : "OFF", sp0);
    double sp = sp0;
    for (int c = 1; c <= cycles; c++) {
        sim.step();
        sp = max_speed(sim.grid());
        if (c % 10 == 0 || !std::isfinite(sp) || sp > 50 * sp0) {
            std::printf("  [clamp %s] cycle %2d: max_speed=%.4f\n", clamp ? "ON " : "OFF", c, sp);
            if (!std::isfinite(sp) || sp > 50 * sp0) {
                std::printf("  [clamp %s] BLEW UP at cycle %d\n", clamp ? "ON " : "OFF", c);
                return sp;
            }
        }
    }
    return sp;
}

int main(int argc, char** argv) {
    int NX        = argc > 1 ? std::atoi(argv[1]) : 48;
    int cycles    = argc > 2 ? std::atoi(argv[2]) : 60;
    double dtfac  = argc > 3 ? std::atof(argv[3]) : 0.25;
    std::printf("=== Inviscid head-on collision, %d^3, %d cycles, dt=%.3g*dx ===\n", NX, cycles,
                dtfac);
    std::printf("--- clamp OFF (expect blow-up) ---\n");
    double off = run(false, NX, cycles, dtfac);
    std::printf("--- clamp ON  (expect bounded) ---\n");
    double on = run(true, NX, cycles, dtfac);

    bool off_blew = !std::isfinite(off) || off > 1e3;
    bool on_ok    = std::isfinite(on) && on < 1e3;
    std::printf("\nResult: clamp OFF final max_speed=%.4g  clamp ON final max_speed=%.4g\n", off, on);
    if (on_ok)
        std::printf("PASS: BFECC clamp keeps the inviscid cycle bounded.%s\n",
                    off_blew ? " (and OFF blew up, as expected)" : "");
    else
        std::printf("FAIL: clamp ON did not stay bounded.\n");
    return on_ok ? 0 : 1;
}

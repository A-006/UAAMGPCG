/**
 * @file test_tgv_mms.cpp
 * @brief Taylor-Green vortex MMS — verifies the unsteady NS solver (convection
 *        + diffusion + projection) is correct via an exact analytic NS solution.
 *
 * TGV on [0,1]^2 with a = 2*pi is an exact solution of the incompressible NS
 * equations: u=-cos(ax)sin(ay)e^{-2 nu a^2 t}, v=sin(ax)cos(ay)e^{-2 nu a^2 t},
 * p=-1/4(cos2ax+cos2ay)e^{-4 nu a^2 t}. We impose the analytic fields as
 * Dirichlet BC (velocity + pressure), march with Chorin projection, and measure
 * the velocity L2 error at time T as the triangle mesh is refined. The measured
 * spatial order of accuracy is the proof the full solver is correct.
 */
#include "../test_utils.h"
#include "unstructured/fvm_poisson.h"
#include "unstructured/ns_solver.h"

#include <cmath>
#include <cstdio>

using namespace ufvm;

int main() {
    test_header("Taylor-Green vortex MMS — unsteady NS (convection+diffusion+projection)");

    const double nu = 0.025;
    const double a  = 2.0 * M_PI;
    auto U          = [&](Vec2 x, double t) {
        return -std::cos(a * x.x) * std::sin(a * x.y) * std::exp(-2 * nu * a * a * t);
    };
    auto V = [&](Vec2 x, double t) {
        return std::sin(a * x.x) * std::cos(a * x.y) * std::exp(-2 * nu * a * a * t);
    };
    auto P = [&](Vec2 x, double t) {
        return -0.25 * (std::cos(2 * a * x.x) + std::cos(2 * a * x.y)) *
               std::exp(-4 * nu * a * a * t);
    };

    auto cg = [](const CSR& A) { return cg_solve(A, A.b, 20000, 1e-12); };

    const double T  = 0.01;
    const double dt = 1e-4; // fixed: keeps Chorin temporal error subdominant

    double prev = 0, last_order = 0;
    bool finite = true;
    for (int n : {16, 32, 64, 128}) {
        PolyMesh m = make_rect_tri(n);
        CSR A      = assemble_laplacian(m, [](Vec2) { return 0.0; });
        NSConfig cfg;
        cfg.nu      = nu;
        cfg.p_outer = 2;

        NSState s = ns_init(m, U, V, P, 0.0);
        NSBoundary bc;
        bc.kind.assign(m.faces.size(), BKind::PressureDirichlet);
        bc.bc_u = U;
        bc.bc_v = V;
        bc.bc_p = P;

        double t   = 0.0;
        int nsteps = (int)std::lround(T / dt);
        for (int k = 0; k < nsteps; ++k) {
            ns_step(m, A, s, bc, cfg, t, dt, cg);
            t += dt;
        }

        double e2 = 0, vol = 0;
        for (int c = 0; c < m.n_cells; ++c) {
            double du = s.u[c] - U(m.centroid[c], t);
            double dv = s.v[c] - V(m.centroid[c], t);
            e2 += (du * du + dv * dv) * m.vol[c];
            vol += m.vol[c];
        }
        double err   = std::sqrt(e2 / vol);
        double order = prev > 0 ? std::log(prev / err) / std::log(2.0) : 0.0;
        std::printf("  n=%3d cells=%6d  vel L2err=%.4e  order=%.3f\n", n, m.n_cells, err, order);
        if (prev > 0)
            last_order = order;
        if (!(err > 0) || std::isnan(err))
            finite = false;
        prev = err;
    }

    check(finite, "velocity errors finite");
    check(last_order > 1.6, "TGV velocity order of accuracy > 1.6 (NS solver is ~2nd-order)");
    return test_summary();
}

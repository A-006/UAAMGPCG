/**
 * @file test_neumann_mms.cpp
 * @brief MMS convergence test for the MIXED Dirichlet/Neumann FVM Poisson
 *        assembly (the capability the cylinder pressure-Poisson needs).
 *
 * Manufactured solution p = cos(pi x) sin(pi y) on [0,1]^2. It satisfies
 * dp/dx = -pi sin(pi x) sin(pi y) = 0 at x=0 and x=1 (exact zero-gradient
 * Neumann), and p = 0 at y=0,1 (Dirichlet). Source for -div grad p = f is
 * f = 2 pi^2 p. Asserts the measured L2 order approaches 2 — proof the mixed
 * Dirichlet/Neumann assembly is correct. Solver-independent (internal CG).
 */
#include "../test_utils.h"
#include "unstructured/fvm_poisson.h"

#include <cmath>
#include <cstdio>

using namespace ufvm;

static double p_exact(Vec2 p) {
    return std::cos(M_PI * p.x) * std::sin(M_PI * p.y);
}
static double source(Vec2 p) {
    return -2.0 * M_PI * M_PI * p_exact(p); // f = lap p = -2 pi^2 p (code's convention)
}
// Neumann on the x = 0 and x = 1 walls, Dirichlet on y = 0,1.
static BCType bc_type(Vec2 c) {
    if (c.x < 1e-9 || c.x > 1.0 - 1e-9)
        return BCType::Neumann;
    return BCType::Dirichlet;
}

static double l2_error(const PolyMesh& m, const std::vector<double>& p) {
    double e2 = 0, v = 0;
    for (int c = 0; c < m.n_cells; ++c) {
        double e = p[c] - p_exact(m.centroid[c]);
        e2 += e * e * m.vol[c];
        v += m.vol[c];
    }
    return std::sqrt(e2 / v);
}

int main() {
    test_header("Mixed Dirichlet/Neumann FVM Poisson — MMS convergence");

    auto cg = [](const CSR& A) { return cg_solve(A, A.b, 20000, 1e-12); };

    double prev = 0, order_last = 0;
    bool all_pos = true;
    for (int n : {16, 32, 64}) {
        PolyMesh m            = make_rect_tri(n);
        std::vector<double> p = solve_poisson(m, source, p_exact, 12, cg, bc_type);
        double err            = l2_error(m, p);
        double order          = prev > 0 ? std::log(prev / err) / std::log(2.0) : 0.0;
        std::printf("  n=%3d cells=%6d  L2err=%.3e  order=%.3f\n", n, m.n_cells, err, order);
        if (prev > 0)
            order_last = order;
        if (err <= 0 || std::isnan(err))
            all_pos = false;
        prev = err;
    }

    check(all_pos, "all L2 errors finite and positive");
    check(order_last > 1.7, "mixed-BC order of accuracy > 1.7 (Neumann assembly is 2nd-order)");
    return test_summary();
}

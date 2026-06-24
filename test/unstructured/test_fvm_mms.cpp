/**
 * @file test_fvm_mms.cpp
 * @brief MMS convergence test for the unstructured FVM Poisson assembly.
 *
 * Manufactured solution p = sin(pi x) sin(pi y) on [0,1]^2 (zero Dirichlet on
 * the boundary), source f = lap p = -2 pi^2 p. Solves on triangle meshes with
 * the internal CG (solver-independent check of the assembly) and asserts the
 * measured L2 order of accuracy approaches 2 — the proof the matrix is right.
 */
#include "../test_utils.h"
#include "unstructured/fvm_poisson.h"

#include <cmath>
#include <cstdio>

using namespace ufvm;

static double p_exact(Vec2 p) {
    return std::sin(M_PI * p.x) * std::sin(M_PI * p.y);
}
static double source(Vec2 p) {
    return -2.0 * M_PI * M_PI * p_exact(p);
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
    test_header("Unstructured FVM Poisson — MMS convergence (triangle mesh)");

    auto cg = [](const CSR& A) { return cg_solve(A, A.b, 20000, 1e-12); };

    double prev = 0, order_last = 0;
    bool all_pos = true;
    for (int n : {16, 32, 64}) {
        PolyMesh m            = make_rect_tri(n);
        std::vector<double> p = solve_poisson(m, source, p_exact, 10, cg);
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
    check(order_last > 1.7, "triangle-mesh order of accuracy > 1.7 (assembly is 2nd-order)");
    return test_summary();
}

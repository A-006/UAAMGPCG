// Verification: the matrix-free multigrid-preconditioned PCG solver.
// Reproduces (in 2D) the paper's Sec.4.3 sinusoidal Poisson study:
//   * solution accuracy ~2nd order (uniform & adaptive)
//   * grid-independent PCG convergence (per-iteration reduction ~constant)
//   * algebraically-consistent (Galerkin) coarsening beats geometric on
//     adaptive grids (fewer iterations / robust) — reproduces Fig.9/14 contrast
//   * matrix-free PCG solution == dense direct solve (independent ground truth)
//   * random/adversarial refinement still converges
#include "semistruct/adaptive_grid_2d.h"
#include "semistruct/poisson_operator_2d.h"
#include "semistruct/multigrid_2d.h"
#include "semistruct/pcg_2d.h"
#include "semistruct/dense_solve.h"
#include "semistruct/sdf_2d.h"
#include "../test_utils.h"
#include <cmath>
#include <cstdio>
#include <random>

using namespace semistruct;

static const double PI = 3.14159265358979323846;
// Neumann-compatible manufactured solution (zero normal derivative on ∂[0,1]²).
static double fexact(double x, double y) { return std::cos(2 * PI * x) * std::cos(2 * PI * y); }
static double forcing(double x, double y) {  // -∇²f
    return 8 * PI * PI * std::cos(2 * PI * x) * std::cos(2 * PI * y);
}
static double zeroMean(std::vector<double>& v) {
    double m = 0; for (double x : v) m += x; m /= v.size();
    for (double& x : v) x -= m; return m;
}

struct SolveOut {
    double err = 0;      // RMS(solution - exact)
    int iters = 0;
    double reduction = 0;  // geometric-mean per-iteration residual reduction
};

static SolveOut solvePoisson(AdaptiveGrid2D& g, Coarsening cm, double tol = 1e-8) {
    g.bc = {BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann};  // pure Neumann (singular)
    PoissonOperator2D op; op.build(g);
    Multigrid2D mg; mg.build(op, cm);
    std::vector<double> b = op.rhs(forcing, [](double, double) { return 0.0; });
    std::vector<double> x;
    auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
    PCGResult R = pcg(op.A, b, x, pre, 200, tol, /*neumann=*/true);
    SolveOut o;
    o.iters = R.iters;
    if (R.res0 > 0 && R.res > 0 && R.iters > 0)
        o.reduction = std::pow(R.res / R.res0, 1.0 / R.iters);
    std::vector<double> ex = op.sample(fexact);
    zeroMean(ex); zeroMean(x);  // compare modulo the Neumann constant
    o.err = op.rmsV(x, ex);
    return o;
}

int main() {
    test_header("Semi-structured multigrid-PCG Poisson solver (2D)");

    // ── 1. Solution accuracy ~2nd order on uniform grids ──
    printf("  uniform solution accuracy:\n");
    double prev = 0, rate_u = 0;
    for (int idx = 0; idx < 4; ++idx) {
        int n = 16 << idx;
        AdaptiveGrid2D g; g.build(1, n, refineUniform());
        SolveOut o = solvePoisson(g, Coarsening::Algebraic);
        if (idx > 0) { rate_u = std::log2(prev / o.err);
            printf("    n=%-4d err=%.3e rate=%.2f iters=%d redux=%.3f\n", n, o.err, rate_u, o.iters, o.reduction); }
        else printf("    n=%-4d err=%.3e iters=%d redux=%.3f\n", n, o.err, o.iters, o.reduction);
        prev = o.err;
    }
    check(rate_u > 1.8, "uniform Poisson solution ~2nd order");

    // ── 2. Grid-independent convergence (uniform) ──
    printf("  grid-independent convergence (uniform):\n");
    int it16, it64;
    {
        AdaptiveGrid2D g; g.build(1, 16, refineUniform());
        it16 = solvePoisson(g, Coarsening::Algebraic).iters;
    }
    {
        AdaptiveGrid2D g; g.build(1, 64, refineUniform());
        it64 = solvePoisson(g, Coarsening::Algebraic).iters;
    }
    printf("    iters: 16x16=%d  64x64=%d\n", it16, it64);
    check(it64 <= it16 + 4, "PCG iteration count grid-independent (uniform)");

    // ── 3. Accuracy on adaptive narrow-band grid ──
    printf("  narrow-band solution accuracy:\n");
    prev = 0; double rate_a = 0;
    for (int idx = 0; idx < 3; ++idx) {
        int n0 = 16 << idx;
        AdaptiveGrid2D g; g.build(3, n0, refineNarrowBand(0.5, 0.5, 0.25, 0.08));
        SolveOut o = solvePoisson(g, Coarsening::Algebraic);
        if (idx > 0) { rate_a = std::log2(prev / o.err);
            printf("    n0=%-4d DOFs=%-7d err=%.3e rate=%.2f iters=%d\n", n0, g.ndof, o.err, rate_a, o.iters); }
        else printf("    n0=%-4d DOFs=%-7d err=%.3e iters=%d\n", n0, g.ndof, o.err, o.iters);
        prev = o.err;
    }
    check(rate_a > 1.5, "narrow-band Poisson solution near 2nd order");

    // ── 4. Algebraic (Galerkin) vs Geometric coarsening on adaptive grid ──
    {
        AdaptiveGrid2D g; g.build(4, 16, refineNarrowBand(0.5, 0.5, 0.25, 0.06));
        SolveOut alg = solvePoisson(g, Coarsening::Algebraic);
        AdaptiveGrid2D g2; g2.build(4, 16, refineNarrowBand(0.5, 0.5, 0.25, 0.06));
        SolveOut geo = solvePoisson(g2, Coarsening::Geometric);
        printf("  adaptive (DOFs=%d): algebraic iters=%d, geometric iters=%d\n",
               g.ndof, alg.iters, geo.iters);
        check(alg.iters <= geo.iters,
              "algebraically-consistent coarsening converges no slower than geometric");
        check(alg.reduction < 0.5, "algebraic per-iteration reduction strong (<0.5)");
    }

    // ── 5. Matrix-free PCG == dense direct solve (independent ground truth) ──
    {
        AdaptiveGrid2D g;
        g.bc = {BC::Dirichlet, BC::Dirichlet, BC::Dirichlet, BC::Dirichlet};
        g.build(2, 6, refineNarrowBand(0.5, 0.5, 0.2, 0.1));  // tiny adaptive grid
        PoissonOperator2D op; op.build(g);
        Multigrid2D mg; mg.build(op, Coarsening::Algebraic);
        std::vector<double> b = op.rhs(forcing, fexact), x, xd;
        auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
        pcg(op.A, b, x, pre, 500, 1e-12, false);
        bool ok = denseSolve(op.A, b, xd);
        check(ok, "dense direct solve succeeds on tiny adaptive grid");
        double diff = 0; for (int i = 0; i < op.A.n; ++i) diff = std::max(diff, std::fabs(x[i] - xd[i]));
        printf("  PCG vs dense direct: max diff = %.3e (DOFs=%d)\n", diff, op.A.n);
        check(diff < 1e-8, "matrix-free PCG solution matches dense direct solve");
    }

    // ── 6. Adversarial random graded refinement still converges ──
    {
        std::mt19937 rng(777);
        std::uniform_real_distribution<double> U(0, 1);
        auto rnd = [&](int, int, int, double, double, double) { return U(rng) < 0.45; };
        AdaptiveGrid2D g; g.build(4, 8, rnd);
        SolveOut o = solvePoisson(g, Coarsening::Algebraic, 1e-8);
        printf("  random-graded (DOFs=%d): iters=%d redux=%.3f err=%.3e\n",
               g.ndof, o.iters, o.reduction, o.err);
        check(o.iters < 60 && o.reduction < 0.7, "random adaptive grid converges robustly");
    }

    return test_summary();
}

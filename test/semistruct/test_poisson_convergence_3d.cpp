// Verification: the 3D matrix-free multigrid-preconditioned PCG solver.
//   * solution accuracy ~2nd order (uniform & adaptive)
//   * grid-independent PCG convergence
//   * algebraically-consistent coarsening beats geometric on adaptive grids
//   * matrix-free PCG == dense direct solve (independent ground truth)
//   * random/adversarial refinement still converges
#include "semistruct/adaptive_grid_3d.h"
#include "semistruct/poisson_operator_3d.h"
#include "semistruct/multigrid_3d.h"
#include "semistruct/pcg_2d.h"   // dimension-agnostic PCG on CSR
#include "semistruct/dense_solve.h"
#include "semistruct/sdf_3d.h"
#include "../test_utils.h"
#include <cmath>
#include <cstdio>
#include <random>

using namespace semistruct;
static const double PI = 3.14159265358979323846;
static double fexact(double x, double y, double z) {
    return std::cos(2*PI*x) * std::cos(2*PI*y) * std::cos(2*PI*z);
}
static double forcing(double x, double y, double z) { return 12*PI*PI*fexact(x,y,z); }
static void zeroMean(std::vector<double>& v) {
    double m = 0; for (double x : v) m += x; m /= v.size(); for (double& x : v) x -= m;
}

struct SolveOut { double err = 0; int iters = 0; double reduction = 0; };

static SolveOut solvePoisson(AdaptiveGrid3D& g, Coarsening cm, double tol = 1e-8) {
    for (auto& b : g.bc) b = BC::Neumann;  // pure Neumann (singular)
    PoissonOperator3D op; op.build(g);
    Multigrid3D mg; mg.build(op, cm);
    std::vector<double> b = op.rhs(forcing, [](double,double,double){ return 0.0; });
    std::vector<double> x;
    auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
    PCGResult R = pcg(op.A, b, x, pre, 200, tol, /*neumann=*/true);
    SolveOut o; o.iters = R.iters;
    if (R.res0 > 0 && R.res > 0 && R.iters > 0) o.reduction = std::pow(R.res/R.res0, 1.0/R.iters);
    std::vector<double> ex = op.sample(fexact);
    zeroMean(ex); zeroMean(x);
    o.err = op.rmsV(x, ex);
    return o;
}

int main() {
    test_header("Semi-structured multigrid-PCG Poisson solver (3D)");

    auto rate = [](double ec, double ef, double ratio) { return std::log(ec/ef)/std::log(ratio); };

    // ── 1. Solution accuracy ~2nd order (uniform) ──
    printf("  uniform solution accuracy:\n");
    double prev = 0, ru = 0; int prevn = 0;
    int ns[3] = {8, 16, 32};
    for (int idx = 0; idx < 3; ++idx) {
        AdaptiveGrid3D g; g.build(1, ns[idx], refineUniform3D());
        SolveOut o = solvePoisson(g, Coarsening::Algebraic);
        if (idx > 0) { ru = rate(prev, o.err, (double)ns[idx]/prevn);
            printf("    n=%-3d err=%.3e rate=%.2f iters=%d redux=%.3f\n", ns[idx], o.err, ru, o.iters, o.reduction); }
        else printf("    n=%-3d err=%.3e iters=%d redux=%.3f\n", ns[idx], o.err, o.iters, o.reduction);
        prev = o.err; prevn = ns[idx];
    }
    check(ru > 1.8, "uniform Poisson solution ~2nd order");

    // ── 2. Grid-independent convergence (uniform) ──
    int it8, it32;
    { AdaptiveGrid3D g; g.build(1, 8, refineUniform3D());  it8 = solvePoisson(g, Coarsening::Algebraic).iters; }
    { AdaptiveGrid3D g; g.build(1, 32, refineUniform3D()); it32 = solvePoisson(g, Coarsening::Algebraic).iters; }
    printf("  grid-independent: iters 8³=%d  32³=%d\n", it8, it32);
    check(it32 <= it8 + 5, "PCG iteration count grid-independent (uniform)");

    // ── 3. Accuracy on adaptive narrow-band sphere ──
    printf("  narrow-band sphere solution accuracy:\n");
    prev = 0; double ra = 0; int prevn0 = 0;
    int n0s[3] = {4, 6, 8};
    for (int idx = 0; idx < 3; ++idx) {
        AdaptiveGrid3D g; g.build(3, n0s[idx], refineNarrowBand3D(0.5,0.5,0.5,0.25,0.1));
        SolveOut o = solvePoisson(g, Coarsening::Algebraic);
        if (idx > 0) { ra = rate(prev, o.err, (double)n0s[idx]/prevn0);
            printf("    n0=%-3d DOFs=%-7d err=%.3e rate=%.2f iters=%d\n", n0s[idx], g.ndof, o.err, ra, o.iters); }
        else printf("    n0=%-3d DOFs=%-7d err=%.3e iters=%d\n", n0s[idx], g.ndof, o.err, o.iters);
        prev = o.err; prevn0 = n0s[idx];
    }
    check(ra > 1.5, "narrow-band sphere Poisson solution near 2nd order");

    // ── 4. Algebraic vs geometric coarsening on adaptive grid ──
    {
        AdaptiveGrid3D g;  g.build(3, 6, refineNarrowBand3D(0.5,0.5,0.5,0.25,0.08));
        SolveOut alg = solvePoisson(g, Coarsening::Algebraic);
        AdaptiveGrid3D g2; g2.build(3, 6, refineNarrowBand3D(0.5,0.5,0.5,0.25,0.08));
        SolveOut geo = solvePoisson(g2, Coarsening::Geometric);
        printf("  adaptive (DOFs=%d): algebraic iters=%d, geometric iters=%d\n", g.ndof, alg.iters, geo.iters);
        check(alg.iters <= geo.iters, "algebraic coarsening converges no slower than geometric");
        check(alg.reduction < 0.6, "algebraic per-iteration reduction strong (<0.6)");
    }

    // ── 5. Matrix-free PCG == dense direct solve (tiny adaptive Dirichlet) ──
    {
        AdaptiveGrid3D g;
        for (auto& b : g.bc) b = BC::Dirichlet;
        g.build(2, 3, refineNarrowBand3D(0.5,0.5,0.5,0.2,0.15));
        PoissonOperator3D op; op.build(g);
        Multigrid3D mg; mg.build(op, Coarsening::Algebraic);
        std::vector<double> b = op.rhs(forcing, fexact), x, xd;
        auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
        pcg(op.A, b, x, pre, 500, 1e-12, false);
        bool ok = denseSolve(op.A, b, xd);
        check(ok, "dense direct solve succeeds on tiny adaptive grid");
        double diff = 0; for (int i = 0; i < op.A.n; ++i) diff = std::max(diff, std::fabs(x[i]-xd[i]));
        printf("  PCG vs dense direct: max diff = %.3e (DOFs=%d)\n", diff, op.A.n);
        check(diff < 1e-7, "matrix-free PCG solution matches dense direct solve");
    }

    // ── 6. Adversarial random graded refinement converges ──
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<double> U(0, 1);
        auto rnd = [&](int, int, int, int, double, double, double, double) { return U(rng) < 0.4; };
        AdaptiveGrid3D g; g.build(3, 4, rnd);
        SolveOut o = solvePoisson(g, Coarsening::Algebraic, 1e-8);
        printf("  random-graded (DOFs=%d): iters=%d redux=%.3f\n", g.ndof, o.iters, o.reduction);
        check(o.iters < 80 && o.reduction < 0.8, "random adaptive grid converges robustly");
    }

    return test_summary();
}

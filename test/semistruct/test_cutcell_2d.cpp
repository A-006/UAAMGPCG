// Verification: cut-cell Poisson (solid obstacle via fluid AREA FRACTIONS, eq.4
// generalisation; Neumann solid + Dirichlet air). Reproduces the paper's Sec.4.4
// cut-cell study in 2D, whose central claim is that ALGEBRAICALLY CONSISTENT
// (Galerkin) coarsening stays robust on cut cells where geometric coarsening
// degrades (Fig.14).
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

static double matrixDiff(const CSR& A, const CSR& B) {
    if (A.n != B.n) return 1e30;
    auto Ma = densify(A), Mb = densify(B);
    double e = 0; for (size_t i = 0; i < Ma.size(); ++i) e = std::max(e, std::fabs(Ma[i] - Mb[i]));
    return e;
}
static void zeroMean(std::vector<double>& v) {
    double m = 0; for (double x : v) m += x; m /= v.size(); for (double& x : v) x -= m;
}

int main() {
    test_header("Cut-cell Poisson (2D)");

    // ── 1. No solid ⇒ buildCut reduces to the plain uniform operator ──
    {
        AdaptiveGrid2D g; g.build(1, 16, refineUniform());
        PoissonOperator2D plain; plain.build(g);
        PoissonOperator2D cut;   cut.buildCut(g, [](double, double) { return 1.0; });  // all fluid
        check(cut.A.n == plain.A.n, "no-solid cut DOF count == plain");
        check_approx(matrixDiff(cut.A, plain.A), 0.0, 1e-12, "no-solid cut operator == plain operator");
    }

    // ── 2. Grid-aligned cut (solid half-plane x<0.5) ⇒ clean Neumann wall, 2nd order ──
    // f = cos(2π(x-0.5)/0.5) cos(2πy) has zero normal derivative on the reduced
    // box [0.5,1]×[0,1] (the cut wall and all outer walls), so the cut system is
    // pure Neumann and the solution should converge 2nd order.
    {
        auto fexact = [](double x, double y) { return std::cos(4 * PI * (x - 0.5)) * std::cos(2 * PI * y); };
        auto forcing = [](double x, double y) { return 20 * PI * PI * std::cos(4 * PI * (x - 0.5)) * std::cos(2 * PI * y); };
        auto solid = [](double x, double) { return x - 0.5; };  // solid where x<0.5
        printf("  grid-aligned cut (solid x<0.5) accuracy:\n");
        double prev = 0, rate = 0;
        for (int idx = 0; idx < 3; ++idx) {
            int n = 16 << idx;
            AdaptiveGrid2D g; g.build(1, n, refineUniform());  // pure Neumann walls
            PoissonOperator2D op; op.buildCut(g, solid);
            Multigrid2D mg; mg.build(op, Coarsening::Algebraic);
            std::vector<double> b = op.rhs(forcing, [](double, double) { return 0.0; }), x;
            auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
            pcg(op.A, b, x, pre, 200, 1e-9, true);
            std::vector<double> ex = op.sample(fexact); zeroMean(ex); zeroMean(x);
            double err = op.rmsV(x, ex);
            if (idx > 0) rate = std::log2(prev / err);
            printf("    n=%-4d fluidDOFs=%-6d err=%.3e rate=%.2f\n", n, op.A.n, err, rate);
            prev = err;
        }
        check(rate > 1.8, "grid-aligned cut solution ~2nd order");
    }

    // ── 3. Sphere obstacle + Dirichlet top: SPD, conservation, algebraic vs geometric ──
    auto sphere = [](double x, double y) { return sdfCircle(x, y, 0.5, 0.5, 0.18); };  // solid disk
    {
        // pure-Neumann cut (all walls Neumann, solid disk) ⇒ A·1 == 0 (conservation)
        AdaptiveGrid2D g; g.build(1, 32, refineUniform());
        PoissonOperator2D op; op.buildCut(g, sphere);
        std::vector<double> ones(op.A.n, 1.0), Aones; op.A.matvec(ones, Aones);
        double m = 0; for (double v : Aones) m = std::max(m, std::fabs(v));
        check(m < 1e-12, "cut operator conserves (A·1==0, pure Neumann)");
        check(symmetryError(op.A) < 1e-12 || op.A.n > 1500, "cut operator symmetric");
    }
    {
        // SPD with Dirichlet top (air), and the Fig.14 robustness comparison.
        printf("  sphere obstacle, algebraic vs geometric coarsening:\n");
        int it_alg64 = 0, it_alg128 = 0;
        for (int n : {32, 64, 128}) {
            AdaptiveGrid2D g;
            g.bc = {BC::Neumann, BC::Neumann, BC::Neumann, BC::Dirichlet};  // top = air (p=0)
            g.build(1, n, refineUniform());
            PoissonOperator2D op; op.buildCut(g, sphere);
            Multigrid2D mga; mga.build(op, Coarsening::Algebraic);
            Multigrid2D mgg; mgg.build(op, Coarsening::Geometric);
            // RHS from a divergence source (downward velocity hitting the obstacle)
            std::vector<double> b = op.rhs([](double x, double y) { return std::sin(3 * x) * std::cos(3 * y); },
                                           [](double, double) { return 0.0; });
            std::vector<double> xa, xg;
            auto pa = [&](const std::vector<double>& r, std::vector<double>& z) { mga.apply(r, z); };
            auto pg = [&](const std::vector<double>& r, std::vector<double>& z) { mgg.apply(r, z); };
            PCGResult Ra = pcg(op.A, b, xa, pa, 300, 1e-8, false);
            PCGResult Rg = pcg(op.A, b, xg, pg, 300, 1e-8, false);
            printf("    n=%-4d fluidDOFs=%-6d  algebraic iters=%d  geometric iters=%d\n",
                   n, op.A.n, Ra.iters, Rg.iters);
            if (n == 64) it_alg64 = Ra.iters;
            if (n == 128) it_alg128 = Ra.iters;
            check(Ra.iters <= Rg.iters, "cut: algebraic converges no slower than geometric");
        }
        check(it_alg128 <= it_alg64 + 6, "cut: algebraic PCG ~grid-independent");
    }

    // ── 4. matrix-free cut PCG == dense direct solve (independent ground truth) ──
    {
        AdaptiveGrid2D g;
        g.bc = {BC::Dirichlet, BC::Dirichlet, BC::Dirichlet, BC::Dirichlet};
        g.build(1, 12, refineUniform());
        PoissonOperator2D op; op.buildCut(g, [](double x, double y) { return sdfCircle(x, y, 0.5, 0.5, 0.15); });
        Multigrid2D mg; mg.build(op, Coarsening::Algebraic);
        std::vector<double> b = op.rhs([](double, double) { return 1.0; }, [](double, double) { return 0.0; }), x, xd;
        auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
        pcg(op.A, b, x, pre, 500, 1e-12, false);
        bool ok = denseSolve(op.A, b, xd);
        double diff = 0; for (int i = 0; i < op.A.n; ++i) diff = std::max(diff, std::fabs(x[i] - xd[i]));
        printf("  cut PCG vs dense: fluidDOFs=%d max diff=%.3e\n", op.A.n, diff);
        check(ok && diff < 1e-8, "matrix-free cut PCG == dense direct solve");
    }

    return test_summary();
}

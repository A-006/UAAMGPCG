// Verification: 3D cut-cell Poisson (solid obstacle via fluid AREA fractions,
// Neumann solid + Dirichlet air). Mirror of test_cutcell_2d. Central claim
// (Sec.4.4 / Fig.14): algebraically consistent (Galerkin) coarsening stays
// grid-independent on cut cells where geometric coarsening degrades.
#include "semistruct/adaptive_grid_3d.h"
#include "semistruct/poisson_operator_3d.h"
#include "semistruct/multigrid_3d.h"
#include "semistruct/pcg_2d.h"  // generic pcg
#include "semistruct/dense_solve.h"
#include "semistruct/sdf_3d.h"
#include "../test_utils.h"
#include <cmath>
#include <cstdio>

using namespace semistruct;
static const double PI = 3.14159265358979323846;

// Sparse symmetric-difference of two CSRs with identical sparsity intent.
static double sparseMatrixDiff(const CSR& A, const CSR& B) {
    if (A.n != B.n) return 1e30;
    double e = 0;
    for (int r = 0; r < A.n; ++r) {
        std::map<int, double> rowB;
        for (int p = B.rowptr[r]; p < B.rowptr[r + 1]; ++p) rowB[B.col[p]] = B.val[p];
        for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; ++p)
            e = std::max(e, std::fabs(A.val[p] - rowB[A.col[p]]));
    }
    return e;
}
static void zeroMean(std::vector<double>& v) {
    double m = 0; for (double x : v) m += x; m /= v.size(); for (double& x : v) x -= m;
}

int main() {
    test_header("Cut-cell Poisson (3D)");

    // ── 1. No solid ⇒ buildCut reduces to the plain uniform operator ──
    {
        AdaptiveGrid3D g; g.build(1, 12, refineUniform3D());
        PoissonOperator3D plain; plain.build(g);
        PoissonOperator3D cut;   cut.buildCut(g, [](double, double, double) { return 1.0; });
        check(cut.A.n == plain.A.n, "no-solid cut DOF count == plain");
        check_approx(sparseMatrixDiff(cut.A, plain.A), 0.0, 1e-12, "no-solid cut operator == plain operator");
    }

    // ── 2. Grid-aligned cut (solid x<0.5) ⇒ clean Neumann wall, 2nd order ──
    {
        auto fexact = [](double x, double y, double z) {
            return std::cos(4 * PI * (x - 0.5)) * std::cos(2 * PI * y) * std::cos(2 * PI * z); };
        auto forcing = [](double x, double y, double z) {
            return 24 * PI * PI * std::cos(4 * PI * (x - 0.5)) * std::cos(2 * PI * y) * std::cos(2 * PI * z); };
        auto solid = [](double x, double, double) { return x - 0.5; };
        printf("  grid-aligned cut (solid x<0.5) accuracy:\n");
        double prev = 0, rate = 0;
        for (int idx = 0; idx < 3; ++idx) {
            int n = 12 + 8 * idx;  // 12,20,28
            AdaptiveGrid3D g; g.build(1, n, refineUniform3D());
            PoissonOperator3D op; op.buildCut(g, solid);
            Multigrid3D mg; mg.build(op, Coarsening::Algebraic);
            std::vector<double> b = op.rhs(forcing, [](double, double, double) { return 0.0; }), x;
            auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
            pcg(op.A, b, x, pre, 300, 1e-9, true);
            std::vector<double> ex = op.sample(fexact); zeroMean(ex); zeroMean(x);
            double err = op.rmsV(x, ex);
            if (idx > 0) rate = std::log(prev / err) / std::log((double)n / (n - 8));
            printf("    n=%-3d fluidDOFs=%-7d err=%.3e rate=%.2f\n", n, op.A.n, err, rate);
            prev = err;
        }
        check(rate > 1.8, "grid-aligned cut solution ~2nd order");
    }

    // ── 3. Sphere obstacle: conservation + algebraic vs geometric (Fig.14) ──
    auto sphere = [](double x, double y, double z) { return sdfSphere(x, y, z, 0.5, 0.5, 0.5, 0.18); };
    {
        AdaptiveGrid3D g; g.build(1, 24, refineUniform3D());
        PoissonOperator3D op; op.buildCut(g, sphere);
        std::vector<double> ones(op.A.n, 1.0), Aones; op.A.matvec(ones, Aones);
        double m = 0; for (double v : Aones) m = std::max(m, std::fabs(v));
        check(m < 1e-12, "cut operator conserves (A·1==0, pure Neumann)");
    }
    {
        printf("  sphere obstacle, algebraic vs geometric coarsening:\n");
        int it_alg_first = 0, it_alg_last = 0;
        int idx = 0;
        for (int n : {16, 24, 32}) {
            AdaptiveGrid3D g;
            g.bc = {BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann, BC::Dirichlet}; // +z air
            g.build(1, n, refineUniform3D());
            PoissonOperator3D op; op.buildCut(g, sphere);
            Multigrid3D mga; mga.build(op, Coarsening::Algebraic);
            Multigrid3D mgg; mgg.build(op, Coarsening::Geometric);
            std::vector<double> b = op.rhs([](double x, double y, double z) { return std::sin(3 * x) * std::cos(3 * y) * std::cos(2 * z); },
                                           [](double, double, double) { return 0.0; });
            std::vector<double> xa, xg;
            auto pa = [&](const std::vector<double>& r, std::vector<double>& z) { mga.apply(r, z); };
            auto pg = [&](const std::vector<double>& r, std::vector<double>& z) { mgg.apply(r, z); };
            PCGResult Ra = pcg(op.A, b, xa, pa, 300, 1e-8, false);
            PCGResult Rg = pcg(op.A, b, xg, pg, 300, 1e-8, false);
            printf("    n=%-3d fluidDOFs=%-7d  algebraic iters=%d  geometric iters=%d\n",
                   n, op.A.n, Ra.iters, Rg.iters);
            check(Ra.iters <= Rg.iters, "cut: algebraic converges no slower than geometric");
            if (idx == 0) it_alg_first = Ra.iters;
            it_alg_last = Ra.iters;
            ++idx;
        }
        check(it_alg_last <= it_alg_first + 6, "cut: algebraic PCG ~grid-independent");
    }

    // ── 4. matrix-free cut PCG == dense direct solve (independent ground truth) ──
    {
        AdaptiveGrid3D g;
        g.bc = {BC::Dirichlet, BC::Dirichlet, BC::Dirichlet, BC::Dirichlet, BC::Dirichlet, BC::Dirichlet};
        g.build(1, 8, refineUniform3D());
        PoissonOperator3D op; op.buildCut(g, [](double x, double y, double z) { return sdfSphere(x, y, z, 0.5, 0.5, 0.5, 0.18); });
        Multigrid3D mg; mg.build(op, Coarsening::Algebraic);
        std::vector<double> b = op.rhs([](double, double, double) { return 1.0; }, [](double, double, double) { return 0.0; }), x, xd;
        auto pre = [&](const std::vector<double>& r, std::vector<double>& z) { mg.apply(r, z); };
        pcg(op.A, b, x, pre, 500, 1e-12, false);
        bool ok = denseSolve(op.A, b, xd);
        double diff = 0; for (int i = 0; i < op.A.n; ++i) diff = std::max(diff, std::fabs(x[i] - xd[i]));
        printf("  cut PCG vs dense: fluidDOFs=%d max diff=%.3e\n", op.A.n, diff);
        check(ok && diff < 1e-8, "matrix-free cut PCG == dense direct solve");
    }

    return test_summary();
}

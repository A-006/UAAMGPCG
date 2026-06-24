// Verification: the composite finite-volume Poisson operator on the
// semi-structured adaptive grid.
//   * uniform operator == independent textbook 5-point Laplacian
//   * symmetry  A == A^T   (random/adversarial refinement included)
//   * discrete conservation  A·1 == 0  for pure Neumann (⇒ flux conservation eq.11)
//   * SPD with a Dirichlet side (lambda_min > 0)
//   * explicit T-junction coupling = 2/3, symmetric
//   * matrix-free Algorithm-3 coarsening == assembled Galerkin (alpha=2)
#include "semistruct/adaptive_grid_2d.h"
#include "semistruct/poisson_operator_2d.h"
#include "semistruct/coarsen_alg3_2d.h"
#include "semistruct/dense_solve.h"
#include "semistruct/sdf_2d.h"
#include "../test_utils.h"
#include <random>

using namespace semistruct;

static double maxAbs(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

static double matrixDiff(const CSR& A, const CSR& B) {
    auto Ma = densify(A), Mb = densify(B);
    double e = 0;
    for (size_t i = 0; i < Ma.size(); ++i) e = std::max(e, std::fabs(Ma[i] - Mb[i]));
    return e;
}

int main() {
    test_header("Semi-structured composite Poisson operator (2D)");

    // ── 1. Uniform operator == textbook 5-point Laplacian ──
    {
        AdaptiveGrid2D g;
        g.build(1, 16, refineUniform());  // single level, 16x16 uniform
        PoissonOperator2D op;
        op.build(g);
        CSR ref = uniformLaplacian5pt(16, g.bc);
        check(op.A.n == 256, "uniform 16x16 has 256 DOFs");
        check_approx(matrixDiff(op.A, ref), 0.0, 1e-12,
                     "matrix-free operator == independent 5-point Laplacian");
    }

    // ── 2. Symmetry + conservation across configurations ──
    auto checkSymmConserv = [&](AdaptiveGrid2D& g, const std::string& tag) {
        PoissonOperator2D op;
        op.build(g);
        check(symmetryError(op.A) < 1e-12, tag + ": A is symmetric");
        std::vector<double> ones(op.A.n, 1.0), Aones;
        op.A.matvec(ones, Aones);
        check(maxAbs(Aones) < 1e-12, tag + ": A·1 == 0 (discrete conservation / Neumann)");
    };
    {
        AdaptiveGrid2D g; g.build(3, 8, refineUniform());      checkSymmConserv(g, "uniform-3lvl");
    }
    {
        AdaptiveGrid2D g; g.build(2, 8, refineLeftHalf());     checkSymmConserv(g, "two-level");
    }
    {
        AdaptiveGrid2D g; g.build(3, 8, refineNarrowBand(0.5, 0.5, 0.25, 0.06));
        checkSymmConserv(g, "narrow-band");
    }
    // adversarial: random graded refinement
    {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> U(0, 1);
        auto rnd = [&](int l, int, int, double, double, double) { return U(rng) < 0.5; };
        AdaptiveGrid2D g; g.build(4, 4, rnd);
        checkSymmConserv(g, "random-graded");
    }

    // ── 3. SPD with a Dirichlet side ──
    {
        AdaptiveGrid2D g;
        g.bc = {BC::Dirichlet, BC::Neumann, BC::Neumann, BC::Neumann};
        g.build(2, 6, refineLeftHalf());
        PoissonOperator2D op; op.build(g);
        double lam = smallestEig(op.A, 80);
        check(lam > 1e-9, "Dirichlet system is SPD (lambda_min > 0)");
    }

    // ── 4. Explicit T-junction coupling = 2/3, symmetric ──
    {
        AdaptiveGrid2D g; g.build(2, 8, refineLeftHalf());
        PoissonOperator2D op; op.build(g);
        auto M = densify(op.A);
        int n = op.A.n;
        // find a coarse leaf (level 0) adjacent to fine leaves (level 1)
        bool found = false; double kappa_seen = 0;
        for (int d = 0; d < n && !found; ++d) {
            if (g.dof_level[d] != 0) continue;
            for (int e = 0; e < n; ++e) {
                if (g.dof_level[e] != 1) continue;
                double aij = M[(size_t)d * n + e];
                if (std::fabs(aij) > 1e-12) {
                    kappa_seen = -aij;  // off-diagonal is -kappa
                    // symmetry partner
                    check_approx(M[(size_t)e * n + d], aij, 1e-12,
                                 "T-junction coupling symmetric");
                    found = true; break;
                }
            }
        }
        check(found, "found a coarse/fine T-junction coupling");
        check_approx(kappa_seen, 2.0 / 3.0, 1e-12, "T-junction kappa == 2/3");
    }

    // ── 5. Matrix-free Algorithm-3 coarsening == assembled Galerkin (uniform) ──
    {
        AdaptiveGrid2D g; g.build(1, 16, refineUniform());
        PoissonOperator2D op; op.build(g);            // 16x16 uniform, pure Neumann
        CompactLevel fine = extractCompact(op.A, 16);
        CompactLevel coarse = coarsenAlg3(fine);      // → 8x8
        CSR alg3 = compactToCSR(coarse);
        // assembled Galerkin with 2x2 aggregation, alpha=2
        std::vector<int> agg(256);
        auto id = [&](int i, int j) { return i + j * 16; };
        for (int j = 0; j < 16; ++j)
            for (int i = 0; i < 16; ++i) agg[id(i, j)] = (i / 2) + (j / 2) * 8;
        CSR gal = galerkin(op.A, agg, 64, 2.0);
        check_approx(matrixDiff(alg3, gal), 0.0, 1e-12,
                     "Alg-3 matrix-free coarsening == Galerkin P^T A P / 2");
    }

    return test_summary();
}

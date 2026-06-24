// Verification: the composite finite-volume Poisson operator (3D).
//   * uniform operator == independent textbook 7-point Laplacian
//   * symmetry A==Aᵀ + discrete conservation A·1==0 (incl. random-graded)
//   * SPD with a Dirichlet side
//   * T-junction coupling == -(1/3)·h_coarse, symmetric
//   * matrix-free Algorithm-3 coarsening == assembled Galerkin (alpha=2)
#include "semistruct/adaptive_grid_3d.h"
#include "semistruct/poisson_operator_3d.h"
#include "semistruct/coarsen_alg3_3d.h"
#include "semistruct/dense_solve.h"
#include "semistruct/sdf_3d.h"
#include "../test_utils.h"
#include <random>

using namespace semistruct;

static double maxAbs(const std::vector<double>& v) {
    double m = 0; for (double x : v) m = std::max(m, std::fabs(x)); return m;
}
// Sparse symmetry error: max |A_ij - A_ji| without densifying.
static double symErrSparse(const CSR& A) {
    double e = 0;
    for (int r = 0; r < A.n; ++r)
        for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; ++p) {
            int c = A.col[p]; double aij = A.val[p], aji = 0;
            for (int q = A.rowptr[c]; q < A.rowptr[c + 1]; ++q)
                if (A.col[q] == r) { aji = A.val[q]; break; }
            e = std::max(e, std::fabs(aij - aji));
        }
    return e;
}
static double matrixDiff(const CSR& A, const CSR& B) {
    auto Ma = densify(A), Mb = densify(B);
    double e = 0; for (size_t i = 0; i < Ma.size(); ++i) e = std::max(e, std::fabs(Ma[i] - Mb[i]));
    return e;
}

int main() {
    test_header("Semi-structured composite Poisson operator (3D)");

    // ── 1. Uniform operator == textbook 7-point Laplacian ──
    {
        AdaptiveGrid3D g; g.build(1, 8, refineUniform3D());  // 8³ uniform
        PoissonOperator3D op; op.build(g);
        CSR ref = uniformLaplacian7pt(8, g.bc);
        check(op.A.n == 512, "uniform 8³ has 512 DOFs");
        check_approx(matrixDiff(op.A, ref), 0.0, 1e-13,
                     "matrix-free operator == independent 7-point Laplacian");
    }

    // ── 2. Symmetry + conservation across configurations ──
    auto checkSC = [&](AdaptiveGrid3D& g, const std::string& tag) {
        PoissonOperator3D op; op.build(g);
        check(symErrSparse(op.A) < 1e-12, tag + ": A is symmetric");
        std::vector<double> ones(op.A.n, 1.0), Aones;
        op.A.matvec(ones, Aones);
        check(maxAbs(Aones) < 1e-12, tag + ": A·1 == 0 (conservation / Neumann)");
    };
    { AdaptiveGrid3D g; g.build(2, 4, refineUniform3D());   checkSC(g, "uniform-2lvl"); }
    { AdaptiveGrid3D g; g.build(2, 6, refineLeftHalf3D());  checkSC(g, "two-level"); }
    { AdaptiveGrid3D g; g.build(2, 6, refineNarrowBand3D(0.5,0.5,0.5,0.25,0.1)); checkSC(g, "narrow-band"); }
    {
        std::mt19937 rng(2024);
        std::uniform_real_distribution<double> U(0, 1);
        auto rnd = [&](int, int, int, int, double, double, double, double) { return U(rng) < 0.5; };
        AdaptiveGrid3D g; g.build(3, 3, rnd); checkSC(g, "random-graded");
    }

    // ── 3. SPD with a Dirichlet side (tiny grid) ──
    {
        AdaptiveGrid3D g;
        g.bc = {BC::Dirichlet, BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann};
        g.build(2, 3, refineLeftHalf3D());
        PoissonOperator3D op; op.build(g);
        double lam = smallestEig(op.A, 60);
        check(lam > 1e-9, "Dirichlet system is SPD (lambda_min > 0)");
    }

    // ── 4. Explicit T-junction coupling = -(1/3) h_coarse, symmetric ──
    {
        AdaptiveGrid3D g; g.build(2, 6, refineLeftHalf3D());
        PoissonOperator3D op; op.build(g);
        double hc = g.lev[0].h, expect = -(1.0 / 3.0) * hc;
        bool found = false; double seen = 0;
        int n = op.A.n;
        for (int d = 0; d < n && !found; ++d) {
            if (g.dof_level[d] != 0) continue;
            for (int p = op.A.rowptr[d]; p < op.A.rowptr[d + 1]; ++p) {
                int e = op.A.col[p];
                if (e == d || g.dof_level[e] != 1) continue;
                seen = op.A.val[p];
                // symmetric partner
                double aji = 0;
                for (int q = op.A.rowptr[e]; q < op.A.rowptr[e + 1]; ++q)
                    if (op.A.col[q] == d) aji = op.A.val[q];
                check_approx(aji, seen, 1e-13, "T-junction coupling symmetric");
                found = true; break;
            }
        }
        check(found, "found a coarse/fine T-junction coupling");
        check_approx(seen, expect, 1e-13, "T-junction coupling == -(1/3) h_coarse");
    }

    // ── 5. Matrix-free Algorithm-3 == assembled Galerkin (uniform 8³ → 4³) ──
    {
        AdaptiveGrid3D g; g.build(1, 8, refineUniform3D());
        PoissonOperator3D op; op.build(g);
        CompactLevel3D fine = extractCompact3D(op.A, 8);
        CSR alg3 = compactToCSR3D(coarsenAlg3_3D(fine));
        std::vector<int> agg(512);
        auto id = [&](int i, int j, int k) { return i + j * 8 + k * 64; };
        for (int k = 0; k < 8; ++k)
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i) agg[id(i, j, k)] = (i/2) + (j/2)*4 + (k/2)*16;
        CSR gal = galerkin(op.A, agg, 64, 2.0);
        check_approx(matrixDiff(alg3, gal), 0.0, 1e-12,
                     "Alg-3 matrix-free coarsening == Galerkin PᵀAP/2");
    }

    return test_summary();
}

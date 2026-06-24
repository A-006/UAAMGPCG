// ─────────────────────────────────────────────────────────────────────────
// Multigrid V-cycle preconditioner on the composite semi-structured system.
//
// The whole composite leaf vector is the finest hierarchy level (A[0] = the
// composite operator). Coarsening uses an octree-informed piecewise-constant
// AGGREGATION: at each step the cells at the current finest scale merge 2x2
// into their parent, while coarser leaves pass through until the coarsening
// front reaches their scale. This is exactly the paper's constant prolongation
// P (P_iJ = 1 if i is a child of J) with R = P^T.
//
// Two ways to form the coarse operator:
//   * ALGEBRAIC (Galerkin)  A_c = P^T A P  — algebraically consistent coarsening
//     (the paper's method; reproduces the fine couplings exactly in uniform
//     regions and accounts for inactive/cut children at the active boundary).
//   * GEOMETRIC              A_c = graph Laplacian of the coarse adjacency
//     (unit coefficients) — the non-conservative baseline ("GMG"). Identical to
//     Galerkin on uniform grids, but inconsistent at adaptive interfaces.
//
// Symmetric Gauss-Seidel smoothing + alpha=1 Galerkin ⇒ SPD preconditioner.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "sparse.h"
#include "adaptive_grid_2d.h"
#include "poisson_operator_2d.h"
#include <vector>
#include <algorithm>

namespace semistruct {

enum class Coarsening { Algebraic, Geometric };

struct Multigrid2D {
    std::vector<CSR> A;
    std::vector<std::vector<int>> agg;  // agg[k] maps level k -> level k+1
    std::vector<int> nC;
    int nu1 = 2, nu2 = 2, nu_coarse = 40;
    int mu = 1;          // cycle index: 1 = V-cycle, 2 = W-cycle
    double beta = 2.0;   // prolongation over-correction (paper: beta=2 for the
                         // constant-prolongation PCG preconditioner)
    Coarsening mode = Coarsening::Algebraic;

    // node coords per fine DOF: (reslevel, i, j) with resolution n0*2^reslevel.
    void build(const PoissonOperator2D& op, Coarsening cmode = Coarsening::Algebraic) {
        mode = cmode;
        A.clear(); agg.clear(); nC.clear();
        A.push_back(op.A);
        // node coordinates at the finest hierarchy level — from the operator's
        // ACTIVE (fluid) DOF list so cut-cell solids are excluded correctly.
        int na = op.A.n;
        std::vector<int> rl(na), ci(na), cj(na);
        for (int d = 0; d < na; ++d) {
            rl[d] = op.node_level[d]; ci[d] = op.node_i[d]; cj[d] = op.node_j[d];
        }
        const int MAXLEV = 40;
        for (int k = 0; k < MAXLEV && A.back().n > 1; ++k) {
            int Lmax = *std::max_element(rl.begin(), rl.end());
            // build aggregation
            std::vector<int> a(A.back().n, -1);
            std::map<long long, int> parentId;
            std::vector<int> nrl, nci, ncj;
            auto key = [](int r, int i, int j) {
                return (((long long)(r + 64)) << 40) ^ ((long long)i << 20) ^ (long long)j;
            };
            for (int m = 0; m < A.back().n; ++m) {
                int pr, pi, pj;
                if (rl[m] == Lmax) { pr = Lmax - 1; pi = ci[m] >> 1; pj = cj[m] >> 1; }
                else { pr = rl[m]; pi = ci[m]; pj = cj[m]; }
                long long kk = key(pr, pi, pj);
                auto it = parentId.find(kk);
                int id;
                if (it == parentId.end()) {
                    id = (int)nrl.size();
                    parentId[kk] = id;
                    nrl.push_back(pr); nci.push_back(pi); ncj.push_back(pj);
                } else id = it->second;
                a[m] = id;
            }
            int nc = (int)nrl.size();
            if (nc >= A.back().n) break;  // no reduction
            CSR Ac = (mode == Coarsening::Algebraic)
                         ? galerkin(A.back(), a, nc, 1.0)
                         : geometricCoarse(A.back(), a, nc);
            agg.push_back(a);
            nC.push_back(nc);
            A.push_back(std::move(Ac));
            rl.swap(nrl); ci.swap(nci); cj.swap(ncj);
        }
    }

    // Geometric coarse operator: unit graph Laplacian on the coarse adjacency.
    static CSR geometricCoarse(const CSR& Af, const std::vector<int>& a, int nc) {
        std::vector<std::map<int, double>> rows(nc);
        for (int r = 0; r < Af.n; ++r) {
            int R = a[r];
            for (int p = Af.rowptr[r]; p < Af.rowptr[r + 1]; ++p) {
                int C = a[Af.col[p]];
                if (C == R) continue;
                if (Af.col[p] == r) continue;
                rows[R][C] = -1.0;  // unit off-diagonal, ignore true magnitude
            }
        }
        std::vector<std::vector<std::pair<int, double>>> off(nc);
        std::vector<double> diagv(nc, 0.0);
        for (int R = 0; R < nc; ++R) {
            double d = 0.0;
            for (auto& e : rows[R]) { off[R].push_back(e); d += -e.second; }
            diagv[R] = d;  // graph Laplacian: diag = sum |offdiag|
        }
        return buildCSR(nc, off, diagv);
    }

    void apply(const std::vector<double>& r, std::vector<double>& z) const {
        z.assign(A[0].n, 0.0);
        vcycle(0, z, r);
    }

    void vcycle(int k, std::vector<double>& x, const std::vector<double>& b) const {
        if (k == (int)A.size() - 1) {
            for (int s = 0; s < nu_coarse; ++s) A[k].sgsSweep(x, b);
            return;
        }
        for (int s = 0; s < nu1; ++s) A[k].sgsSweep(x, b);
        std::vector<double> Ax, res(A[k].n);
        A[k].matvec(x, Ax);
        for (int i = 0; i < A[k].n; ++i) res[i] = b[i] - Ax[i];
        // restrict R = P^T (sum children)
        std::vector<double> bc(nC[k], 0.0);
        for (int i = 0; i < A[k].n; ++i) bc[agg[k][i]] += res[i];
        std::vector<double> xc(nC[k], 0.0);
        for (int c = 0; c < mu; ++c) vcycle(k + 1, xc, bc);
        // prolong P (inject) with over-correction beta
        for (int i = 0; i < A[k].n; ++i) x[i] += beta * xc[agg[k][i]];
        for (int s = 0; s < nu2; ++s) A[k].sgsSweep(x, b);
    }
};

}  // namespace semistruct

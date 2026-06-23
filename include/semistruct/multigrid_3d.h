// ─────────────────────────────────────────────────────────────────────────
// Multigrid V/μ-cycle preconditioner (3D). Same composite-grid octree-aggregation
// Galerkin scheme as the 2D version (see multigrid_2d.h): A[0] = composite
// operator; at each step cells at the current finest scale merge 2×2×2 into their
// parent, coarser leaves pass through. β=2 over-correction, symmetric GS.
// Reuses the Coarsening enum and the geometric (unit graph-Laplacian) baseline.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "sparse.h"
#include "multigrid_2d.h"  // Coarsening enum + Multigrid2D::geometricCoarse + generic vcycle helpers
#include "poisson_operator_3d.h"
#include <vector>
#include <algorithm>
#include <map>
#include <array>

namespace semistruct {

struct Multigrid3D {
    std::vector<CSR> A;
    std::vector<std::vector<int>> agg;
    std::vector<int> nC;
    int nu1 = 2, nu2 = 2, nu_coarse = 40;
    int mu = 1;
    double beta = 2.0;
    Coarsening mode = Coarsening::Algebraic;

    void build(const PoissonOperator3D& op, Coarsening cmode = Coarsening::Algebraic) {
        mode = cmode;
        A.clear(); agg.clear(); nC.clear();
        A.push_back(op.A);
        // node coords from the operator's ACTIVE (fluid) DOF list → cut-cell solids excluded.
        int na = op.A.n;
        std::vector<int> rl(na), ci(na), cj(na), ck(na);
        for (int d = 0; d < na; ++d) {
            rl[d] = op.node_level[d]; ci[d] = op.node_i[d]; cj[d] = op.node_j[d]; ck[d] = op.node_k[d];
        }
        const int MAXLEV = 60;
        for (int it = 0; it < MAXLEV && A.back().n > 1; ++it) {
            int Lmax = *std::max_element(rl.begin(), rl.end());
            std::vector<int> a(A.back().n, -1);
            std::map<std::array<int, 4>, int> parentId;
            std::vector<int> nrl, nci, ncj, nck;
            for (int m = 0; m < A.back().n; ++m) {
                int pr, pi, pj, pk;
                if (rl[m] == Lmax) { pr = Lmax - 1; pi = ci[m] >> 1; pj = cj[m] >> 1; pk = ck[m] >> 1; }
                else { pr = rl[m]; pi = ci[m]; pj = cj[m]; pk = ck[m]; }
                std::array<int, 4> kk{pr, pi, pj, pk};
                auto it2 = parentId.find(kk);
                int id;
                if (it2 == parentId.end()) {
                    id = (int)nrl.size();
                    parentId[kk] = id;
                    nrl.push_back(pr); nci.push_back(pi); ncj.push_back(pj); nck.push_back(pk);
                } else id = it2->second;
                a[m] = id;
            }
            int nc = (int)nrl.size();
            if (nc >= A.back().n) break;
            CSR Ac = (mode == Coarsening::Algebraic)
                         ? galerkin(A.back(), a, nc, 1.0)
                         : Multigrid2D::geometricCoarse(A.back(), a, nc);
            agg.push_back(a);
            nC.push_back(nc);
            A.push_back(std::move(Ac));
            rl.swap(nrl); ci.swap(nci); cj.swap(ncj); ck.swap(nck);
        }
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
        std::vector<double> bc(nC[k], 0.0);
        for (int i = 0; i < A[k].n; ++i) bc[agg[k][i]] += res[i];
        std::vector<double> xc(nC[k], 0.0);
        for (int c = 0; c < mu; ++c) vcycle(k + 1, xc, bc);
        for (int i = 0; i < A[k].n; ++i) x[i] += beta * xc[agg[k][i]];
        for (int s = 0; s < nu2; ++s) A[k].sgsSweep(x, b);
    }
};

}  // namespace semistruct

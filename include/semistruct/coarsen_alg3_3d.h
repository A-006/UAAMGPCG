// ─────────────────────────────────────────────────────────────────────────
// Matrix-free algebraically-consistent coarsening — 3D Algorithm 3 ("Coarsen").
// Compact per-cell representation (diag d, neg-x/-y/-z couplings cmx,cmy,cmz) of
// a uniform level → coarse level satisfying A^{l-1} = (1/alpha) Pᵀ A^l P, α=2.
// Verified against the assembled Galerkin operator on uniform grids.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "sparse.h"
#include <vector>

namespace semistruct {

struct CompactLevel3D {
    int n = 0;
    std::vector<double> d, cmx, cmy, cmz;  // each n³
    inline int idx(int i, int j, int k) const { return i + j * n + k * n * n; }
};

inline CompactLevel3D extractCompact3D(const CSR& A, int n) {
    CompactLevel3D c;
    c.n = n;
    size_t N = (size_t)n * n * n;
    c.d.assign(N, 0.0); c.cmx.assign(N, 0.0); c.cmy.assign(N, 0.0); c.cmz.assign(N, 0.0);
    auto id = [&](int i, int j, int k) { return i + j * n + k * n * n; };
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                int r = id(i, j, k);
                c.d[r] = A.diag[r];
                for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; ++p) {
                    int col = A.col[p];
                    if (i > 0 && col == id(i - 1, j, k)) c.cmx[r] = A.val[p];
                    if (j > 0 && col == id(i, j - 1, k)) c.cmy[r] = A.val[p];
                    if (k > 0 && col == id(i, j, k - 1)) c.cmz[r] = A.val[p];
                }
            }
    return c;
}

inline CompactLevel3D coarsenAlg3_3D(const CompactLevel3D& f) {
    const double ia = 1.0 / 2.0;  // 1/alpha, alpha=2
    int nf = f.n, nc = nf / 2;
    CompactLevel3D c;
    c.n = nc;
    size_t N = (size_t)nc * nc * nc;
    c.d.assign(N, 0.0); c.cmx.assign(N, 0.0); c.cmy.assign(N, 0.0); c.cmz.assign(N, 0.0);
    auto fid = [&](int i, int j, int k) { return i + j * nf + k * nf * nf; };
    for (int K = 0; K < nc; ++K)
        for (int J = 0; J < nc; ++J)
            for (int I = 0; I < nc; ++I) {
                int cI = I + J * nc + K * nc * nc;
                double dC = 0, cmxC = 0, cmyC = 0, cmzC = 0;
                int active = 0;
                for (int dk = 0; dk < 2; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            int i = 2 * I + di, j = 2 * J + dj, k = 2 * K + dk;
                            double dijk = f.d[fid(i, j, k)];
                            if (dijk != 0.0) { active++; dC += ia * dijk; }
                            if (di == 1) {
                                if (dijk != 0.0 && f.d[fid(i - 1, j, k)] != 0.0)
                                    dC += 2.0 * ia * f.cmx[fid(i, j, k)];
                            } else cmxC += ia * f.cmx[fid(i, j, k)];
                            if (dj == 1) {
                                if (dijk != 0.0 && f.d[fid(i, j - 1, k)] != 0.0)
                                    dC += 2.0 * ia * f.cmy[fid(i, j, k)];
                            } else cmyC += ia * f.cmy[fid(i, j, k)];
                            if (dk == 1) {
                                if (dijk != 0.0 && f.d[fid(i, j, k - 1)] != 0.0)
                                    dC += 2.0 * ia * f.cmz[fid(i, j, k)];
                            } else cmzC += ia * f.cmz[fid(i, j, k)];
                        }
                if (active == 0) dC = 0;
                c.d[cI] = dC; c.cmx[cI] = cmxC; c.cmy[cI] = cmyC; c.cmz[cI] = cmzC;
            }
    return c;
}

inline CSR compactToCSR3D(const CompactLevel3D& c) {
    int n = c.n, N = n * n * n;
    auto id = [&](int i, int j, int k) { return i + j * n + k * n * n; };
    std::vector<std::vector<std::pair<int, double>>> off(N);
    std::vector<double> diagv(N, 0.0);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                int r = id(i, j, k);
                diagv[r] = c.d[r];
                if (i > 0 && c.cmx[r] != 0.0) {
                    off[r].push_back({id(i - 1, j, k), c.cmx[r]});
                    off[id(i - 1, j, k)].push_back({r, c.cmx[r]});
                }
                if (j > 0 && c.cmy[r] != 0.0) {
                    off[r].push_back({id(i, j - 1, k), c.cmy[r]});
                    off[id(i, j - 1, k)].push_back({r, c.cmy[r]});
                }
                if (k > 0 && c.cmz[r] != 0.0) {
                    off[r].push_back({id(i, j, k - 1), c.cmz[r]});
                    off[id(i, j, k - 1)].push_back({r, c.cmz[r]});
                }
            }
    return buildCSR(N, off, diagv);
}

}  // namespace semistruct

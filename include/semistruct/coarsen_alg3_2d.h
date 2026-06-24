// ─────────────────────────────────────────────────────────────────────────
// Matrix-free algebraically-consistent coarsening — 2D adaptation of Algorithm 3
// ("Coarsen") of the paper. Operates on the compact per-cell representation
// (diagonal d, negative-x coupling cmx, negative-y coupling cmy) of a uniform
// level and produces the coarse level's compact coefficients, satisfying the
// Galerkin principle A^{l-1} = (1/alpha) P^T A^l P with alpha = 2.
//
// Sign convention (matches eq.4 and our operator): d > 0, cmx/cmy < 0.
// This file is used by the verification suite to show the matrix-free coarsening
// reproduces the assembled Galerkin operator exactly in uniform regions.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "sparse.h"
#include <vector>

namespace semistruct {

// Compact coefficient field for a uniform n x n level.
struct CompactLevel {
    int n = 0;
    std::vector<double> d;    // diagonal, size n*n
    std::vector<double> cmx;  // coupling to (i-1,j), size n*n (0 if i==0 or inactive)
    std::vector<double> cmy;  // coupling to (i,j-1), size n*n
    inline int idx(int i, int j) const { return i + j * n; }
};

// Extract the compact representation from the CSR of a UNIFORM n x n grid.
inline CompactLevel extractCompact(const CSR& A, int n) {
    CompactLevel c;
    c.n = n;
    c.d.assign((size_t)n * n, 0.0);
    c.cmx.assign((size_t)n * n, 0.0);
    c.cmy.assign((size_t)n * n, 0.0);
    auto id = [&](int i, int j) { return i + j * n; };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int r = id(i, j);
            c.d[r] = A.diag[r];
            for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; ++p) {
                int col = A.col[p];
                if (i > 0 && col == id(i - 1, j)) c.cmx[r] = A.val[p];
                if (j > 0 && col == id(i, j - 1)) c.cmy[r] = A.val[p];
            }
        }
    return c;
}

// Algorithm 3 (2D), alpha = 2. Coarse level is (n/2) x (n/2).
inline CompactLevel coarsenAlg3(const CompactLevel& f) {
    const double alpha = 2.0;
    const double ia = 1.0 / alpha;
    int nf = f.n, nc = nf / 2;
    CompactLevel c;
    c.n = nc;
    c.d.assign((size_t)nc * nc, 0.0);
    c.cmx.assign((size_t)nc * nc, 0.0);
    c.cmy.assign((size_t)nc * nc, 0.0);
    auto fid = [&](int i, int j) { return i + j * nf; };
    for (int J = 0; J < nc; ++J)
        for (int I = 0; I < nc; ++I) {
            int cI = I + J * nc;
            double dC = 0, cmxC = 0, cmyC = 0;
            int active = 0;
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) {
                    int i = 2 * I + di, j = 2 * J + dj;
                    double dij = f.d[fid(i, j)];
                    if (dij != 0.0) { active++; dC += ia * dij; }
                    // x-direction
                    if (di == 1) {
                        double dim1 = f.d[fid(i - 1, j)];
                        if (dij != 0.0 && dim1 != 0.0) dC += (2.0 * ia) * f.cmx[fid(i, j)];
                    } else {
                        cmxC += ia * f.cmx[fid(i, j)];
                    }
                    // y-direction
                    if (dj == 1) {
                        double djm1 = f.d[fid(i, j - 1)];
                        if (dij != 0.0 && djm1 != 0.0) dC += (2.0 * ia) * f.cmy[fid(i, j)];
                    } else {
                        cmyC += ia * f.cmy[fid(i, j)];
                    }
                }
            if (active == 0) dC = 0;
            c.d[cI] = dC;
            c.cmx[cI] = cmxC;
            c.cmy[cI] = cmyC;
        }
    return c;
}

// Assemble a CSR from a compact uniform level (for comparison).
inline CSR compactToCSR(const CompactLevel& c) {
    int n = c.n, N = n * n;
    auto id = [&](int i, int j) { return i + j * n; };
    std::vector<std::vector<std::pair<int, double>>> off(N);
    std::vector<double> diagv(N, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int r = id(i, j);
            diagv[r] = c.d[r];
            if (i > 0 && c.cmx[r] != 0.0) {
                off[r].push_back({id(i - 1, j), c.cmx[r]});
                off[id(i - 1, j)].push_back({r, c.cmx[r]});  // symmetric partner
            }
            if (j > 0 && c.cmy[r] != 0.0) {
                off[r].push_back({id(i, j - 1), c.cmy[r]});
                off[id(i, j - 1)].push_back({r, c.cmy[r]});
            }
        }
    return buildCSR(N, off, diagv);
}

}  // namespace semistruct

// ─────────────────────────────────────────────────────────────────────────
// Tiny dense direct solver (Gaussian elimination with partial pivoting) used
// ONLY as an independent ground truth for small (<~1000 DOF) non-singular
// systems in the verification tests. No external dependency.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "sparse.h"
#include <vector>
#include <cmath>

namespace semistruct {

// Dense densify of a CSR.
inline std::vector<double> densify(const CSR& A) {
    std::vector<double> M((size_t)A.n * A.n, 0.0);
    for (int r = 0; r < A.n; ++r)
        for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; ++p)
            M[(size_t)r * A.n + A.col[p]] = A.val[p];
    return M;
}

// Solve A x = b by dense Gaussian elimination. Returns false if singular.
inline bool denseSolve(const CSR& A, const std::vector<double>& b, std::vector<double>& x) {
    int n = A.n;
    std::vector<double> M = densify(A);
    std::vector<double> rhs = b;
    auto at = [&](int r, int c) -> double& { return M[(size_t)r * n + c]; };
    for (int c = 0; c < n; ++c) {
        int piv = c;
        double best = std::fabs(at(c, c));
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(at(r, c)) > best) { best = std::fabs(at(r, c)); piv = r; }
        if (best < 1e-14) return false;
        if (piv != c) {
            for (int cc = 0; cc < n; ++cc) std::swap(at(c, cc), at(piv, cc));
            std::swap(rhs[c], rhs[piv]);
        }
        double d = at(c, c);
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            double f = at(r, c) / d;
            if (f == 0.0) continue;
            for (int cc = c; cc < n; ++cc) at(r, cc) -= f * at(c, cc);
            rhs[r] -= f * rhs[c];
        }
    }
    x.assign(n, 0.0);
    for (int r = 0; r < n; ++r) x[r] = rhs[r] / at(r, r);
    return true;
}

// Check symmetry of a CSR (max |A_ij - A_ji|).
inline double symmetryError(const CSR& A) {
    auto M = densify(A);
    double e = 0.0;
    for (int r = 0; r < A.n; ++r)
        for (int c = 0; c < A.n; ++c)
            e = std::max(e, std::fabs(M[(size_t)r * A.n + c] - M[(size_t)c * A.n + r]));
    return e;
}

// Smallest eigenvalue estimate via inverse power iteration (needs non-singular).
// Used to confirm SPD (lambda_min > 0).
inline double smallestEig(const CSR& A, int iters = 200) {
    int n = A.n;
    std::vector<double> v(n, 1.0), w;
    double lambda = 0.0;
    for (int it = 0; it < iters; ++it) {
        if (!denseSolve(A, v, w)) return -1.0;  // singular
        double nrm = norm2(w);
        for (double& x : w) x /= nrm;
        // Rayleigh quotient with A
        std::vector<double> Aw;
        A.matvec(w, Aw);
        lambda = dot(w, Aw);
        v = w;
    }
    return lambda;
}

}  // namespace semistruct

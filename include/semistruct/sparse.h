// ─────────────────────────────────────────────────────────────────────────
// Minimal sparse linear-algebra used by the semi-structured solver: CSR matrix,
// matvec, symmetric Gauss-Seidel smoothing, and the Galerkin triple product
// A_c = (1/alpha) P^T A P for a piecewise-constant (aggregation) prolongation.
// Self-contained, double precision. Kept deliberately simple — correctness and
// verifiability over speed.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include <vector>
#include <map>
#include <cmath>
#include <cstddef>

namespace semistruct {

struct CSR {
    int n = 0;
    std::vector<int> rowptr;     // n+1
    std::vector<int> col;        // nnz
    std::vector<double> val;     // nnz
    std::vector<double> diag;    // n (cached diagonal)

    void finalizeDiag() {
        diag.assign(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int p = rowptr[r]; p < rowptr[r + 1]; ++p)
                if (col[p] == r) diag[r] = val[p];
    }

    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        y.assign(n, 0.0);
        for (int r = 0; r < n; ++r) {
            double s = 0.0;
            for (int p = rowptr[r]; p < rowptr[r + 1]; ++p) s += val[p] * x[col[p]];
            y[r] = s;
        }
    }

    // One forward + one backward Gauss-Seidel sweep (symmetric GS, SPD smoother).
    void sgsSweep(std::vector<double>& x, const std::vector<double>& b) const {
        for (int r = 0; r < n; ++r) gsUpdate(r, x, b);
        for (int r = n - 1; r >= 0; --r) gsUpdate(r, x, b);
    }

    inline void gsUpdate(int r, std::vector<double>& x, const std::vector<double>& b) const {
        if (diag[r] == 0.0) return;
        double s = b[r];
        for (int p = rowptr[r]; p < rowptr[r + 1]; ++p)
            if (col[p] != r) s -= val[p] * x[col[p]];
        x[r] = s / diag[r];
    }
};

// Build a CSR from per-row coupling lists. `offdiag[r]` lists (col, value) pairs
// (the off-diagonal entries, already signed), `diagv[r]` is the diagonal.
inline CSR buildCSR(int n, const std::vector<std::vector<std::pair<int, double>>>& offdiag,
                    const std::vector<double>& diagv) {
    CSR A;
    A.n = n;
    A.rowptr.assign(n + 1, 0);
    for (int r = 0; r < n; ++r) A.rowptr[r + 1] = A.rowptr[r] + 1 + (int)offdiag[r].size();
    int nnz = A.rowptr[n];
    A.col.resize(nnz);
    A.val.resize(nnz);
    for (int r = 0; r < n; ++r) {
        int p = A.rowptr[r];
        A.col[p] = r;
        A.val[p] = diagv[r];
        ++p;
        for (auto& e : offdiag[r]) { A.col[p] = e.first; A.val[p] = e.second; ++p; }
    }
    A.finalizeDiag();
    return A;
}

// Galerkin coarse operator A_c = (1/alpha) P^T A P, where P is a piecewise
// constant aggregation: agg[i] = coarse index of fine DOF i, nC coarse DOFs.
inline CSR galerkin(const CSR& A, const std::vector<int>& agg, int nC, double alpha) {
    std::vector<std::map<int, double>> rows(nC);
    for (int r = 0; r < A.n; ++r) {
        int R = agg[r];
        if (R < 0) continue;
        for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; ++p) {
            int C = agg[A.col[p]];
            if (C < 0) continue;
            rows[R][C] += A.val[p];
        }
    }
    CSR Ac;
    Ac.n = nC;
    Ac.rowptr.assign(nC + 1, 0);
    for (int R = 0; R < nC; ++R) Ac.rowptr[R + 1] = Ac.rowptr[R] + (int)rows[R].size();
    int nnz = Ac.rowptr[nC];
    Ac.col.resize(nnz);
    Ac.val.resize(nnz);
    double inv = 1.0 / alpha;
    for (int R = 0; R < nC; ++R) {
        int p = Ac.rowptr[R];
        for (auto& e : rows[R]) { Ac.col[p] = e.first; Ac.val[p] = inv * e.second; ++p; }
    }
    Ac.finalizeDiag();
    return Ac;
}

// Vector helpers.
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
inline double norm2(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }
inline void axpy(double a, const std::vector<double>& x, std::vector<double>& y) {
    for (size_t i = 0; i < x.size(); ++i) y[i] += a * x[i];
}

}  // namespace semistruct

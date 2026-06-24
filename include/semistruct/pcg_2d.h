// ─────────────────────────────────────────────────────────────────────────
// Preconditioned Conjugate Gradient on the composite leaf vector (Alg. 1 of the
// paper). The preconditioner is any callable  apply(r, z): z ≈ A^{-1} r.
// For a pure-Neumann (singular) system the residual and iterate are projected
// to zero mean each step so PCG stays in the range of A.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "sparse.h"
#include <vector>
#include <functional>

namespace semistruct {

struct PCGResult {
    int iters = 0;
    double res0 = 0, res = 0;
    std::vector<double> residual_history;  // ||r||_2 after each iteration
};

inline void projectZeroMean(std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m += x;
    m /= (double)v.size();
    for (double& x : v) x -= m;
}

// Solve A x = b. precond(r,z) applies the preconditioner. If `neumann`, project
// to zero-mean (singular consistent system).
inline PCGResult pcg(const CSR& A, const std::vector<double>& b, std::vector<double>& x,
                     const std::function<void(const std::vector<double>&, std::vector<double>&)>& precond,
                     int max_iter, double rel_tol, bool neumann) {
    int n = A.n;
    x.assign(n, 0.0);
    std::vector<double> r = b, z(n), p(n), Ap(n);
    if (neumann) projectZeroMean(r);
    PCGResult R;
    R.res0 = norm2(r);
    if (R.res0 == 0.0) return R;
    precond(r, z);
    if (neumann) projectZeroMean(z);
    p = z;
    double rz = dot(r, z);
    for (int it = 0; it < max_iter; ++it) {
        A.matvec(p, Ap);
        if (neumann) projectZeroMean(Ap);
        double pAp = dot(p, Ap);
        double alpha = rz / pAp;
        axpy(alpha, p, x);
        axpy(-alpha, Ap, r);
        double rn = norm2(r);
        R.residual_history.push_back(rn);
        R.iters = it + 1;
        R.res = rn;
        if (rn <= rel_tol * R.res0) break;
        precond(r, z);
        if (neumann) projectZeroMean(z);
        double rz_new = dot(r, z);
        double beta = rz_new / rz;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }
    if (neumann) projectZeroMean(x);
    return R;
}

// Plain CG (identity preconditioner) convenience wrapper.
inline PCGResult cg(const CSR& A, const std::vector<double>& b, std::vector<double>& x,
                    int max_iter, double rel_tol, bool neumann) {
    return pcg(A, b, x, [](const std::vector<double>& r, std::vector<double>& z) { z = r; },
               max_iter, rel_tol, neumann);
}

}  // namespace semistruct

#include "../test_common.h"
#include "../test_utils.h"
#include <algorithm>

// ── V-Cycle (from solver/amgpcg.cpp) ──
static void v_cycle(std::vector<Grid*>& x_lv, std::vector<Grid*>& b_lv, int level) {
    Grid& x = *x_lv[level];
    const Grid& b = *b_lv[level];
    int nl = (int)x_lv.size();
    if (level == nl - 1) { rbgs_smooth(x, b, 20); return; }
    rbgs_smooth(x, b, 2);
    Grid r(x.N);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++) r(i,j) = b(i,j) - Ax_at(x, i, j);
    Grid r_c = restrict(r);
    Grid& b_c = *b_lv[level+1];
    for (int i = 1; i <= b_c.N; i++)
        for (int j = 1; j <= b_c.N; j++) b_c(i,j) = r_c(i,j);
    Grid& x_c = *x_lv[level+1];
    std::fill(x_c.v.begin(), x_c.v.end(), 0.0);
    v_cycle(x_lv, b_lv, level+1);
    Grid corr = prolongate(x_c, x.N, 1.0);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++) x(i,j) += corr(i,j);
    rbgs_smooth(x, b, 2);
}

// ── AMGPCG solver ──
static Grid amgpcg_solve(const Grid& b, int max_iter, double tol, int& out_iters) {
    int N = b.N;
    auto levels = build_levels(N);
    std::vector<Grid> s_x, s_b;
    for (int n : levels) { s_x.emplace_back(n); s_b.emplace_back(n); }
    int nl = (int)levels.size();
    std::vector<Grid*> xp, bp;
    for (int i = 0; i < nl; i++) { xp.push_back(&s_x[i]); bp.push_back(&s_b[i]); }

    Grid x(N), r(N), z(N);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) r(i,j) = b(i,j);

    auto precondition = [&](Grid& zz, const Grid& rr) {
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) s_b[0](i,j) = rr(i,j);
        for (int l = 0; l < nl; l++) std::fill(s_x[l].v.begin(), s_x[l].v.end(), 0.0);
        v_cycle(xp, bp, 0);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) zz(i,j) = s_x[0](i,j);
    };

    precondition(z, r);
    Grid p = z;
    double rsold = dot(r, z);

    for (int k = 0; k < max_iter; k++) {
        Grid Ap = matvec(p);
        double alpha = rsold / dot(p, Ap);
        axpy(alpha, p, x); axpy(-alpha, Ap, r);
        double rnorm = std::sqrt(dot(r, r));
        if (rnorm < tol) { out_iters = k + 1; return x; }
        precondition(z, r);
        double rsnew = dot(r, z);
        double beta = rsnew / rsold;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) p(i,j) = z(i,j) + beta * p(i,j);
        rsold = rsnew;
    }
    out_iters = max_iter;
    return x;
}

int main() {
    test_header("AMGPCG Unit Tests");

    // Test 1: AMGPCG accuracy (sin)
    {
        int N = 32;
        Grid b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * b.h), sy = std::sin(M_PI * j * b.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }
        int iters = 0;
        Grid x = amgpcg_solve(b, 30, 1e-8, iters);
        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err, std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 1e-2, "AMGPCG sin error < 1e-2");
        check(iters <= 30, "AMGPCG converges in <= 30 iters");
    }

    // Test 2: AMGPCG vs CG on polynomial RHS
    {
        int N = 64;
        Grid b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double xi = i*b.h, yj = j*b.h;
                b(i,j) = 2.0*(xi*(1-xi) + yj*(1-yj));
            }
        // Unpreconditioned CG
        Grid x_cg(N), r(N), p(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) r(i,j) = b(i,j);
        p = r;
        double rsold = dot(r, r);
        int cg_iters = 500;
        for (int k = 0; k < 500; k++) {
            Grid Ap = matvec(p);
            double pAp = dot(p, Ap);
            if (pAp < 1e-15) break;
            double alpha = rsold / pAp;
            axpy(alpha, p, x_cg); axpy(-alpha, Ap, r);
            if (std::sqrt(dot(r,r)) < 1e-8) { cg_iters = k+1; break; }
            double beta = dot(r,r) / rsold;
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++) p(i,j) = r(i,j) + beta * p(i,j);
            rsold = dot(r,r);
        }
        int amg_iters = 0;
        amgpcg_solve(b, 30, 1e-8, amg_iters);
        check(amg_iters < cg_iters/2, "AMGPCG < 1/2 CG its (AMG="+std::to_string(amg_iters)+", CG="+std::to_string(cg_iters)+")");
    }

    // Test 3: Grid-independent convergence
    {
        int it32 = 0, it64 = 0;
        {
            Grid b(32);
            for (int i = 1; i <= 32; i++)
                for (int j = 1; j <= 32; j++) {
                    double sx = std::sin(M_PI*i*b.h), sy = std::sin(M_PI*j*b.h);
                    b(i,j) = 2.0*M_PI*M_PI*sx*sy;
                }
            amgpcg_solve(b, 30, 1e-8, it32);
        }
        {
            Grid b(64);
            for (int i = 1; i <= 64; i++)
                for (int j = 1; j <= 64; j++) {
                    double sx = std::sin(M_PI*i*b.h), sy = std::sin(M_PI*j*b.h);
                    b(i,j) = 2.0*M_PI*M_PI*sx*sy;
                }
            amgpcg_solve(b, 30, 1e-8, it64);
        }
        check(std::abs(it64-it32) <= 5, "iterations nearly independent of N");
    }

    // Test 4: Polynomial RHS
    {
        int N = 32;
        Grid b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double xi = i*b.h, yj = j*b.h;
                b(i,j) = 2.0*(xi*(1-xi) + yj*(1-yj));
            }
        int iters = 0;
        Grid x = amgpcg_solve(b, 30, 1e-8, iters);
        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double xi = i*x.h, yj = j*x.h;
                max_err = std::max(max_err, std::abs(x(i,j) - xi*(1-xi)*yj*(1-yj)));
            }
        check(max_err < 2e-3, "AMGPCG polynomial within discretization error");
    }

    return test_summary();
}

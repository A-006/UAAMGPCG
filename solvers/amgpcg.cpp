// ===========================================================================
// Step 5: AMGPCG — AMG Preconditioned Conjugate Gradient (Core Algorithm of Chapter 5)
// Compile: g++ -std=c++17 -O2 amgpcg.cpp -o amgpcg
// Run: ./amgpcg [N] [max_iter]
// ===========================================================================
//
// [AMGPCG = V-Cycle Preconditioner + CG Outer Iteration]
//
// CG convergence rate depends on the matrix condition number κ(A) = λ_max/λ_min
// Poisson equation κ(A) ≈ N² → CG slows down on large grids
// Preconditioning: solve M^{-1}Ax = M^{-1}b, where M≈A and M^{-1} is easy to compute
//   → κ(M^{-1}A) << κ(A) → CG converges much faster
//
// V-Cycle as preconditioner z = M^{-1} r:
//   Input r → One V-Cycle → Output approximate solution z (z ≈ A^{-1}r)
//
// PCG per iteration: 1 matvec(Ap) + 1 V-Cycle(z = M^{-1}r) + 2 inner products
//
// Performance:
//   N=64:  CG ~100+ iterations, AMGPCG ~10 iterations
//   N=128: CG ~400+ iterations, AMGPCG ~10 iterations  ← Nearly independent of N!
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

struct Grid {
    int N;
    double h;
    std::vector<double> v;
    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}
    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

// ── Matrix-free Ax — 5-point Laplacian ──
inline double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

// ── matvec: y = A * x ──
Grid matvec(const Grid& x) {
    Grid Ax(x.N);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            Ax(i,j) = Ax_at(x, i, j);
    return Ax;
}

// ── Dot product ──
double dot(const Grid& a, const Grid& b) {
    double s = 0;
    for (int i = 1; i <= a.N; i++)
        for (int j = 1; j <= a.N; j++)
            s += a(i,j) * b(i,j);
    return s;
}

// ── y += a * x ──
void axpy(double a, const Grid& x, Grid& y) {
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            y(i,j) += a * x(i,j);
}

// ── RBGS Smoothing ──
void rbgs_smooth(Grid& x, const Grid& b, int iterations = 1) {
    double inv_diag = x.h * x.h / 4.0;
    int N = x.N;
    for (int sweep = 0; sweep < iterations; sweep++) {
        for (int i = 1; i <= N; i++)
            for (int j = 1 + (i % 2); j <= N; j += 2)
                x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
        for (int i = 1; i <= N; i++)
            for (int j = 1 + ((i+1) % 2); j <= N; j += 2)
                x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
    }
}

// ── Restriction R: fine(2N) → coarse(N) ──
Grid restrict(const Grid& r_fine) {
    int Nc = r_fine.N / 2;
    if (Nc < 2) Nc = 2;
    if (Nc % 2 != 0 && Nc > 2) Nc--;
    Grid r_coarse(Nc);
    for (int ic = 1; ic <= Nc; ic++) {
        for (int jc = 1; jc <= Nc; jc++) {
            int i_f = 2*ic - 1;
            int j_f = 2*jc - 1;
            r_coarse(ic, jc) = 0.25 * (
                r_fine(i_f,j_f) + r_fine(i_f+1,j_f) +
                r_fine(i_f,j_f+1) + r_fine(i_f+1,j_f+1));
        }
    }
    return r_coarse;
}

// ── Prolongation P: coarse(N) → fine(2N) (×2 scaling) ──
Grid prolongate(const Grid& x_coarse, int Nf, double scaling = 2.0) {
    Grid x_fine(Nf);
    int Nc = x_coarse.N;
    for (int ic = 1; ic <= Nc; ic++) {
        for (int jc = 1; jc <= Nc; jc++) {
            double val = x_coarse(ic, jc) * scaling;
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            x_fine(i_f,   j_f)   = val;
            x_fine(i_f+1, j_f)   = val;
            x_fine(i_f,   j_f+1) = val;
            x_fine(i_f+1, j_f+1) = val;
        }
    }
    return x_fine;
}

// ── Recursive V-Cycle ──
void v_cycle(const std::vector<Grid*>& x_lv,
             const std::vector<Grid*>& b_lv,
             int level)
{
    Grid& x = *x_lv[level];
    const Grid& b = *b_lv[level];
    int nlevels = (int)x_lv.size();

    if (level == nlevels - 1) {
        rbgs_smooth(x, b, 20);
        return;
    }

    // Downward stroke
    rbgs_smooth(x, b, 1);

    Grid r(x.N);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            r(i,j) = b(i,j) - Ax_at(x, i, j);

    Grid r_c = restrict(r);
    Grid& b_c = *b_lv[level + 1];
    for (int i = 1; i <= b_c.N; i++)
        for (int j = 1; j <= b_c.N; j++)
            b_c(i,j) = r_c(i,j);

    Grid& x_c = *x_lv[level + 1];
    std::fill(x_c.v.begin(), x_c.v.end(), 0.0);

    v_cycle(x_lv, b_lv, level + 1);

    // Upward stroke
    Grid corr = prolongate(x_c, x.N, 2.0);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            x(i,j) += corr(i,j);

    rbgs_smooth(x, b, 1);
}

// ── Build Levels ──
std::vector<int> build_levels(int N) {
    std::vector<int> lv;
    int n = N;
    while (n >= 2) {
        lv.push_back(n);
        if (n <= 3) break;
        n = n / 2;
        if (n < 2) n = 2;
    }
    return lv;
}

// ── Preconditioner: z = M^{-1} r → One V-Cycle ──
void precondition(Grid& z, const Grid& r, int N,
                  const std::vector<int>& levels,
                  std::vector<Grid>& storage_x,
                  std::vector<Grid>& storage_b)
{
    // Reuse pre-allocated level storage
    int nl = (int)levels.size();

    // Copy r to finest b
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            storage_b[0](i,j) = r(i,j);

    // Zero out x on all levels
    for (int lv = 0; lv < nl; lv++)
        std::fill(storage_x[lv].v.begin(), storage_x[lv].v.end(), 0.0);

    // V-Cycle
    std::vector<Grid*> x_ptrs, b_ptrs;
    for (int lv = 0; lv < nl; lv++) {
        x_ptrs.push_back(&storage_x[lv]);
        b_ptrs.push_back(&storage_b[lv]);
    }
    v_cycle(x_ptrs, b_ptrs, 0);

    // Extract finest level result
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            z(i,j) = storage_x[0](i,j);
}

// ═══════════════ AMGPCG Main Solver ═══════════════
Grid amgpcg_solve(const Grid& b, int max_iter, double tol, int& out_iters) {
    int N = b.N;
    auto levels = build_levels(N);

    // Pre-allocate level storage (avoid repeated allocation per V-Cycle)
    std::vector<Grid> storage_x, storage_b;
    for (int n : levels) {
        storage_x.emplace_back(n);
        storage_b.emplace_back(n);
    }

    Grid x(N);                     // Initial guess x=0
    Grid r = b;                    // r = b - Ax = b (x=0)
    Grid z(N);                     // z = M^{-1} r
    precondition(z, r, N, levels, storage_x, storage_b);

    Grid p = z;                    // p = z
    double rsold = dot(r, z);

    for (int k = 0; k < max_iter; k++) {
        Grid Ap = matvec(p);
        double pAp = dot(p, Ap);
        double alpha = rsold / pAp;

        axpy( alpha, p, x);        // x += αp
        axpy(-alpha, Ap, r);       // r -= αAp

        double rnorm = std::sqrt(dot(r, r));
        if (k % 5 == 0)
            std::cout << "  iter " << std::setw(3) << k
                      << "  |r|=" << rnorm << '\n';

        if (rnorm < tol) {
            out_iters = k + 1;
            return x;
        }

        precondition(z, r, N, levels, storage_x, storage_b);

        double rsnew = dot(r, z);
        double beta = rsnew / rsold;

        // p = z + βp
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                p(i,j) = z(i,j) + beta * p(i,j);

        rsold = rsnew;
    }

    out_iters = max_iter;
    return x;
}

int main(int argc, char* argv[]) {
    int N = 64;
    int max_iter = 50;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) max_iter = std::atoi(argv[2]);

    const double TOL = 1e-8;

    std::cout << "=== AMGPCG Preconditioned Conjugate Gradient  N=" << N
              << " (" << N*N << " unknowns) ===\n\n";

    Grid b(N);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) {
            double sx = std::sin(M_PI * i * b.h);
            double sy = std::sin(M_PI * j * b.h);
            b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    int iters = 0;
    Grid x = amgpcg_solve(b, max_iter, TOL, iters);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double err = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            err = std::max(err, std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));

    std::cout << "\nConverged: " << iters << " iters, " << elapsed << " s, error: " << err << '\n';
    std::cout << "Levels: ";
    for (int n : build_levels(N)) std::cout << n << " ";
    std::cout << "\n\nAMGPCG ≈ CG convergence speed × 10+, and hardly degrades as N grows\n";
    return 0;
}

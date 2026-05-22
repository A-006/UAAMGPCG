// ===========================================================================
// Step 3: CG (Conjugate Gradient)
// Compile: g++ -std=c++17 -O2 cg.cpp -o cg
// Run:     ./cg [N]
// ===========================================================================
//
// 【Why Jacobi/RBGS is not enough — bottleneck of local methods】
// Jacobi and RBGS update only one point per round, looking only at itself and neighbors
//   → High-frequency (local) errors are eliminated quickly
//   → Low-frequency (global) errors barely move — e.g. a "large wave" error where the
//     entire region is biased high
//   The larger the grid, the more low-frequency errors → convergence degrades sharply as N grows
//
// 【CG's idea: choose the globally optimal search direction each time】
// Instead of updating point-by-point, advance along a "search direction" in the N²-dimensional space.
// The key is: each new direction p_k is A-conjugate to all previous directions p_0...p_{k-1}
//
//   A-conjugate: p_i^T A p_j = 0  (i≠j)
//
// What does this mean? Each direction is traversed only once, never doing redundant work
// in the same direction.
// n A-conjugate directions span the entire space → theoretical exact convergence in n steps
// (n = number of unknowns)
// In practice, tens to hundreds of steps are enough.
//
// ===========================================================================
// 【CG algorithm — only 1 matrix-vector multiply + 2 inner products per round】
// ===========================================================================
//
// x₀ = 0                         initial guess (arbitrary)
// r₀ = b - Ax₀                   residual (measurement of error)
// p₀ = r₀                        first search direction = residual direction (steepest descent)
//
// for k = 0,1,2,...:
//   αₖ = (rₖ·rₖ) / (pₖ·Apₖ)     optimal step length along pₖ
//   x_{k+1} = xₖ + αₖ·pₖ         update solution
//   r_{k+1} = rₖ - αₖ·Apₖ        update residual (note: Apₖ can be reused!)
//   βₖ = (r_{k+1}·r_{k+1})/(rₖ·rₖ)
//   p_{k+1} = r_{k+1} + βₖ·pₖ    next search direction (A-conjugate correction)
//
// Two key computations:
//   matvec(p): matrix-vector product, using 5-point stencil → matrix-free
//   dot(a,b):  vector inner product, only interior grid points
//
// ===========================================================================
// 【CG vs Jacobi/RBGS — essential difference】
// ===========================================================================
// Jacobi/RBGS: x_{i,j}^{new} = x_{i,j} + ω*(b_{i,j} - Ax_{i,j})
//              Each point is corrected independently, no global coordination
//
// CG:         Each time choose a direction p, walk along p to the exact minimum in that direction
//             Directions are A-conjugate to each other → each direction is traversed only once,
//             never going back
//
// Cost: CG needs inner products (global communication) per round, while Jacobi does not
//       But CG requires far fewer rounds than Jacobi → shorter total time
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

struct Grid {
    int N;
    double h;
    std::vector<double> v;
    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}
    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

// ── A*v — 5-point Laplacian ──
// Returns a new Grid, does not modify input
Grid matvec(const Grid& v) {
    int N = v.N;
    double inv_h2 = 1.0 / (v.h * v.h);
    Grid Av(N);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            Av(i,j) = (4*v(i,j) - v(i+1,j) - v(i-1,j) - v(i,j+1) - v(i,j-1)) * inv_h2;
    return Av;
}

// ── Inner product — interior points only ──
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

// ── y = a * x + y  (same as axpy, another usage pattern in CG) ──
// In CG, p = r + beta * p uses this: first scale old p, then add r
void xpay(const Grid& x, double a, Grid& y) {
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            y(i,j) = x(i,j) + a * y(i,j);
}

#ifndef TEST_MODE
int main(int argc, char* argv[]) {
    int N = 64;
    if (argc > 1) N = std::atoi(argv[1]);

    const double TOL   = 1e-8;
    const int MAX_ITER = 5000;

    std::cout << "=== CG Conjugate Gradient  N=" << N
              << " (" << N*N << " unknowns)  tol=" << TOL << " ===\n\n";

    Grid x(N);   // solution
    Grid b(N);   // right-hand side

    // Known solution u = x(1-x)·y(1-y)  (not an eigenfunction, CG needs multiple rounds)
    // f = -Δu = 2x(1-x) + 2y(1-y)
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) {
            double xi = i * x.h, yj = j * x.h;
            b(i,j) = 2.0 * (xi*(1-xi) + yj*(1-yj));
        }

    // ═══════════════ CG main loop ═══════════════
    auto t0 = std::chrono::high_resolution_clock::now();

    // Initialize: r = b - A*0 = b,  p = r
    Grid r = b;        // residual (copy of b)
    Grid p = r;        // search direction
    double rsold = dot(r, r);         // (r₀, r₀)
    double r0_norm = std::sqrt(rsold); // ||r₀||, used for displaying relative residual

    for (int k = 0; k < MAX_ITER; k++) {
        // ── α = (r·r) / (p·Ap) ──
        Grid Ap = matvec(p);              // matrix-vector product (most expensive operation)
        double pAp = dot(p, Ap);          // (p, Ap)
        double alpha = rsold / pAp;       // optimal step length along p

        // ── x = x + α*p,  r = r - α*Ap ──
        axpy(alpha, p, x);               // x += α * p
        axpy(-alpha, Ap, r);             // r -= α * Ap  (reuse Ap computed above!)

        // ── Check convergence ──
        double rsnew = dot(r, r);
        if (k % 10 == 0)
            std::cout << "iter " << std::setw(4) << k
                      << "  |r|/|r0| = " << std::sqrt(rsnew) / r0_norm << '\n';

        if (std::sqrt(rsnew) < TOL) {  // absolute residual < tol
            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();

            double err = 0;
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++) {
                    double xi = i * x.h, yj = j * x.h;
                    err = std::max(err, std::abs(x(i,j) - xi*(1-xi)*yj*(1-yj)));
                }

            std::cout << "\nConverged: " << k << " iterations, " << s << " s, error " << err << '\n';
            return 0;
        }

        // ── β = (r_new·r_new) / (r_old·r_old) ──
        double beta = rsnew / rsold;

        // ── p = r + β*p (new direction = residual + conjugate correction of old direction) ──
        // First scale old p, then add r_new
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                p(i,j) = r(i,j) + beta * p(i,j);

        rsold = rsnew;
    }

    std::cout << "Did not converge\n";
    return 1;
}
#endif

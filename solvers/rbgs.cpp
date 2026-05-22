// ===========================================================================
// Step 2: RBGS (Red-Black Gauss-Seidel) Smoother
// Build: g++ -std=c++17 -O2 rbgs.cpp -o rbgs
// Run:   ./rbgs [N]
// ===========================================================================
//
// Why Jacobi is Not Enough
// Jacobi updates all points synchronously with old values each round
//   → information propagation speed = 1 grid/round
// For an N=64 grid, the error at the center takes 32 rounds to reach
//   the boundary → tens of thousands of rounds needed
//
// Gauss-Seidel: Use New Values Immediately After Computation
// Point-by-point update: when updating point (i,j), (i-1,j) and (i,j-1)
//   already hold new values
// Faster information propagation → roughly twice the convergence speed
// But the problem is: sequential dependency prevents parallelization
//
// RBGS: Red-Black Coloring Solves Parallelization
// Under the 5-point Laplacian stencil, each point only connects to its
//   up/down/left/right neighbors
// Checkerboard coloring: (i+j) even=red, odd=black
//   → Red points only connect to black points, black points only connect
//     to red points
//   → No two red points are adjacent → can be updated simultaneously!
//   → After red points are updated, all black points can also be updated
//     simultaneously!
//
// Effect: convergence close to Gauss-Seidel, but fully parallel within
//   each color
// This is the smoother for each level in Algorithm 3 (V-Cycle) of the
//   paper
//
// ===========================================================================
// RBGS Update Formula
// ===========================================================================
// Same as Jacobi: x_{i,j} += D⁻¹ (b_{i,j} - (Ax)_{i,j})
//                 x_{i,j} += (h²/4) * [b_{i,j} - (4x_{ij}-Σneighbors)/h²]
//
// Difference: Jacobi's Ax reads all old values, RBGS's Ax reads the
//   current latest values (mixed old and new)
// Jacobi needs x_new backup array, RBGS modifies in-place
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

// Matrix-free Ax — identical to Jacobi
double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

// ── RBGS One Sweep (modify x in-place) ──
void rbgs_sweep(Grid& x, const Grid& b) {
    double inv_diag = x.h * x.h / 4.0;   // h²/4
    int N = x.N;

    // Phase 1: Red points — (i+j) even
    // Red points are not adjacent to each other, update order within
    //   the same phase does not matter
    for (int i = 1; i <= N; i++)
        for (int j = 1 + (i % 2); j <= N; j += 2)
            x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));

    // Phase 2: Black points — (i+j) odd
    // At this point red points have been updated, black points' Ax will
    //   use the new red values → faster convergence
    for (int i = 1; i <= N; i++)
        for (int j = 1 + ((i+1) % 2); j <= N; j += 2)
            x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
}

// ── Jacobi (for comparison) ──
void jacobi_sweep(Grid& x, const Grid& b) {
    double inv_diag = x.h * x.h / 4.0;
    int N = x.N;
    std::vector<double> x_new(x.v.size());

    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            x_new[i*(N+2)+j] = x(i,j) + inv_diag * (b(i,j) - Ax_at(x, i, j));
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            x(i,j) = x_new[i*(N+2)+j];
}

// Compute maximum residual
double residual_max(const Grid& x, const Grid& b) {
    double rmax = 0;
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            rmax = std::max(rmax, std::abs(b(i,j) - Ax_at(x, i, j)));
    return rmax;
}

int main(int argc, char* argv[]) {
    int N = 64;
    if (argc > 1) N = std::atoi(argv[1]);

    const double TOL   = 1e-6;
    const int MAX_ITER = 50000;

    std::cout << "=== RBGS vs Jacobi Convergence Comparison  N=" << N
              << " (" << N*N << " unknowns) ===\n\n";

    // ═══════════════ Jacobi (Reference) ═══════════════
    {
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        auto t0 = std::chrono::high_resolution_clock::now();
        int it = 0;
        for (; it < MAX_ITER; it++) {
            jacobi_sweep(x, b);
            if (it % 200 == 0 && residual_max(x, b) < TOL) break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Jacobi  " << std::setw(6) << it << " iters  "
                  << s << " s  res=" << residual_max(x, b) << '\n';
    }

    // ═══════════════ RBGS ═══════════════
    {
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        auto t0 = std::chrono::high_resolution_clock::now();
        int it = 0;
        for (; it < MAX_ITER; it++) {
            rbgs_sweep(x, b);
            if (it % 200 == 0 && residual_max(x, b) < TOL) break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "RBGS   " << std::setw(6) << it << " iters  "
                  << s << " s  res=" << residual_max(x, b) << '\n';
    }

    std::cout << "\nRBGS is about 2× faster than Jacobi, and does not need the x_new backup array\n";
    std::cout << "But as a standalone solver it is still too slow — the real use of RBGS is as a smoother in multigrid (Step 4)\n";
    return 0;
}

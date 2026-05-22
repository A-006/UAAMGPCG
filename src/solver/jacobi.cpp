// ===========================================================================
// Step 1: Jacobi iterative solution of 2D Poisson equation
// Compile: g++ -std=c++17 -O2 jacobi.cpp -o jacobi
// Run: ./jacobi [N]          (default N=64)
// ===========================================================================
//
// Problem: Solve on the unit square [0,1]×[0,1]
//      -Δu = f       (Poisson equation)
//      u = 0 on boundary   (Dirichlet boundary condition)
//
// Known solution for verification: u(x,y) = sin(πx)·sin(πy)
// Substituting yields RHS:   f(x,y) = 2π²·sin(πx)·sin(πy)
//
// ===========================================================================
// [From continuous to discrete — derivation of the 5-point Laplacian stencil]
// ===========================================================================
//
// Central difference approximation of the second derivative (h is grid spacing):
//   u''(x) ≈ [u(x+h) - 2u(x) + u(x-h)] / h²
//   The error of this formula is O(h²), i.e., second-order accuracy
//
// 2D Laplace operator Δu = ∂²u/∂x² + ∂²u/∂y², on a uniform grid:
//
//   Δu|_{i,j} ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / h²     ← x direction
//             + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / h²     ← y direction
//
//   Combined:  (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
//
//   Add the negative sign (Poisson is -Δu = f):
//
//   (-Δu)|_{i,j} ≈ (4u_{i,j} - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1}) / h²
//
//   This is the formula in Ax_at() — 5-point stencil, using only itself and 4 neighbors
//
//   Structure of matrix A:
//     diagonal =  4/h²    ← coefficient of itself
//     neighbors = -1/h²    ← coefficients of up/down/left/right
//     rest     =  0       ← non-adjacent points have no direct connection
//
// ===========================================================================
// [Jacobi iteration formula]
// ===========================================================================
//
//   Ax = b  →  Decompose A into D + (A-D), D=diagonal
//   Dx = b - (A-D)x
//   x^{new} = D⁻¹(b - A x^{old})         ← Jacobi iteration
//
//   D⁻¹ = h²/4  (reciprocal of diagonal: 1 ÷ (4/h²))
//
//   Each grid point: x_{i,j}^{new} = x_{i,j}^{old} + (h²/4)·[b_{i,j} - (Ax)^{old}_{i,j}]
//                                                   └────── residual ──────────┘
//
//   Feature: all points are updated synchronously with old values → inherently parallel
//   Drawback: information propagates only 1 grid cell per round → very slow convergence on large grids
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "solver/jacobi.h"

// ── Grid: 2D array (N+2)×(N+2), flattened into 1D storage ──
// Index: (i,j) → i*(N+2)+j
// Boundary rows/cols (i=0, N+1, j=0, N+1) store boundary conditions, do not participate in iteration
struct Grid {
    int N;
    double h;
    std::vector<double> v;

    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}

    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

#ifndef TEST_MODE
int main(int argc, char* argv[]) {
    int N = 64;
    if (argc > 1) N = std::atoi(argv[1]);

    const double TOL   = 1e-6;
    const int MAX_ITER = 50000;
    double inv_h2 = 1.0 / ((N+1.0)*(N+1.0));

    Grid x(N), b(N), u(N);

    // True solution u=sin(πx)sin(πy) → f=-Δu=2π²·sin(πx)·sin(πy)
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) {
            double sx = std::sin(M_PI * i * x.h);
            double sy = std::sin(M_PI * j * x.h);
            b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            u(i,j) = sx * sy;
        }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Matrix-free Ax: 5-point Laplacian with Dirichlet BC
    auto Ax = [&](int i, int j) -> double {
        return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
    };
    // D⁻¹ = h²/4, uniform
    auto inv_diag = [&](int, int) -> double { return x.h * x.h / 4.0; };

    jacobi_solve(x.v, b.v, Ax, inv_diag, N, N, N+2, MAX_ITER, TOL);

    auto t1 = std::chrono::high_resolution_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();

    double err = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            err = std::max(err, std::abs(x(i,j) - u(i,j)));

    std::cout << "N=" << N << "  time=" << s << " s  error=" << err << '\n';
    return 0;
}
#endif

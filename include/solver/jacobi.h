#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

// Generic matrix-free Jacobi iteration: x_new = x + D⁻¹ * (b - A*x)
//
// AxFunc:  double(int i, int j)  — returns (A*x)_{i,j}
// DiagFunc: double(int i, int j) — returns D⁻¹_{i,j}
//
// Interior points: i ∈ [1,N], j ∈ [1,M]. Boundaries (0 and N+1/M+1) are not iterated.

template<typename AxFunc, typename DiagFunc>
void jacobi_iterate(std::vector<double>& x, const std::vector<double>& x_copy,
                    const std::vector<double>& b,
                    AxFunc&& Ax, DiagFunc&& inv_diag,
                    int N, int M, int stride) {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            int idx = i * stride + j;
            x[idx] = x_copy[idx] + inv_diag(i, j) * (b[idx] - Ax(i, j));
        }
    }
}

// Full Jacobi solver: repeats jacobi_iterate until convergence or max_iter.
// Returns number of iterations used. Stops early if ||b - Ax||_max < tol.
template<typename AxFunc, typename DiagFunc>
int jacobi_solve(std::vector<double>& x, const std::vector<double>& b,
                 AxFunc&& Ax, DiagFunc&& inv_diag,
                 int N, int M, int stride, int max_iter, double tol) {
    std::vector<double> xn = x;
    for (int iter = 0; iter < max_iter; iter++) {
        jacobi_iterate(x, xn, b,
                       std::forward<AxFunc>(Ax),
                       std::forward<DiagFunc>(inv_diag),
                       N, M, stride);
        xn = x;

        if (iter % 200 == 0) {
            double rmax = 0;
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= M; j++) {
                    int idx = i * stride + j;
                    rmax = std::max(rmax, std::abs(b[idx] - Ax(i, j)));
                }
            if (rmax < tol) return iter + 1;
        }
    }
    return max_iter;
}

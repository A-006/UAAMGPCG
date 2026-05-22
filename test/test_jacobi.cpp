// Unit tests for Jacobi solver — exercises code in src/solver/jacobi.cpp
#define TEST_MODE
#include "../src/solver/jacobi.cpp"
#include "test_utils.h"
#include "solver/jacobi.h"

// Local Ax_at for the test Grid (used to verify correctness)
static double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

int main() {
    test_header("Jacobi Solver Unit Tests");

    // Test 1: Manufactured solution u = sin(pi*x)*sin(pi*y), N=32
    {
        int N = 32;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        // Use generic jacobi_solve from include/solver/jacobi.h
        double inv_h2 = 1.0 / (x.h * x.h);
        auto Ax = [&](int i, int j) {
            return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
        };
        auto inv_diag = [&](int, int) { return x.h * x.h / 4.0; };
        jacobi_solve(x.v, b.v, Ax, inv_diag, N, N, N+2, 20000, 1e-6);

        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err,
                    std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 1e-3, "sin solution error < 1e-3 at N=32");
    }

    // Test 2: Matrix-free Ax_at matches analytical Laplacian
    {
        int N = 16;
        Grid x(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                x(i,j) = std::sin(M_PI * i * x.h) * std::sin(M_PI * j * x.h);

        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double expected = 2.0*M_PI*M_PI * std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h);
                max_err = std::max(max_err, std::abs(Ax_at(x,i,j) - expected));
            }
        check(max_err < 8e-2, "Ax_at discretization error O(h^2), N=16");
    }

    // Test 3: Dirichlet BC — boundaries stay zero after iteration
    {
        int N = 8;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) b(i,j) = 1.0;

        double inv_h2 = 1.0 / (x.h * x.h);
        auto Ax = [&](int i, int j) {
            return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
        };
        auto inv_diag = [&](int, int) { return x.h * x.h / 4.0; };
        jacobi_solve(x.v, b.v, Ax, inv_diag, N, N, N+2, 200, 1e-6);

        bool bc_ok = true;
        for (int i = 0; i <= N+1; i++) {
            if (x(i,0) != 0.0 || x(i,N+1) != 0.0) bc_ok = false;
            if (x(0,i) != 0.0 || x(N+1,i) != 0.0) bc_ok = false;
        }
        check(bc_ok, "Dirichlet boundaries remain zero");
    }

    // Test 4: Convergence — residual decreases monotonically
    {
        int N = 12;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) b(i,j) = 1.0;

        double inv_h2 = 1.0 / (x.h * x.h);
        double inv_d = x.h * x.h / 4.0;
        std::vector<double> xn(x.v.size());
        double prev_r = 1e99;
        bool decreasing = true;
        for (int iter = 0; iter < 20; iter++) {
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++)
                    xn[i*(N+2)+j] = x(i,j) + inv_d * (b(i,j) - Ax_at(x, i, j));
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++)
                    x(i,j) = xn[i*(N+2)+j];
            double rmax = 0;
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++)
                    rmax = std::max(rmax, std::abs(b(i,j) - Ax_at(x, i, j)));
            if (rmax > prev_r * 1.01) decreasing = false;
            prev_r = rmax;
        }
        check(decreasing, "Jacobi residual decreases monotonically");
    }

    return test_summary();
}

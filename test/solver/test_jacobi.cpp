#include "../test_common.h"
#include "../test_utils.h"

// Simple inline Jacobi solver for test Grid (Dirichlet BC, uniform h)
static void jacobi_solve(Grid& x, const Grid& b, int max_iter, double tol) {
    double inv_d = x.h * x.h / 4.0;
    std::vector<double> xn(x.v.size());
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 1; i <= x.N; i++)
            for (int j = 1; j <= x.N; j++)
                xn[i*(x.N+2)+j] = x(i,j) + inv_d * (b(i,j) - Ax_at(x, i, j));
        for (int i = 1; i <= x.N; i++)
            for (int j = 1; j <= x.N; j++)
                x(i,j) = xn[i*(x.N+2)+j];
        if (iter % 200 == 0 && residual_max(x, b) < tol) break;
    }
}

int main() {
    test_header("Jacobi Solver Unit Tests");

    // Test 1: Manufactured solution
    {
        int N = 32;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI*i*x.h), sy = std::sin(M_PI*j*x.h);
                b(i,j) = 2.0*M_PI*M_PI*sx*sy;
            }
        jacobi_solve(x, b, 20000, 1e-6);
        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err, std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 1e-3, "sin solution error < 1e-3 at N=32");
    }

    // Test 2: Ax_at discretization error
    {
        int N = 16; Grid x(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                x(i,j) = std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h);
        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err, std::abs(Ax_at(x,i,j) - 2.0*M_PI*M_PI*std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 8e-2, "Ax_at error O(h^2), N=16");
    }

    // Test 3: Dirichlet BC stay zero
    {
        int N = 8; Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) b(i,j) = 1.0;
        jacobi_solve(x, b, 200, 1e-6);
        bool ok = true;
        for (int i = 0; i <= N+1; i++) {
            if (x(i,0)||x(i,N+1)||x(0,i)||x(N+1,i)) ok = false;
        }
        check(ok, "Dirichlet boundaries remain zero");
    }

    // Test 4: Residual decreases monotonically
    {
        int N = 12; Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) b(i,j) = 1.0;
        double inv_d = x.h*x.h/4.0, prev_r = 1e99;
        std::vector<double> xn(x.v.size());
        bool dec = true;
        for (int iter = 0; iter < 20; iter++) {
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++)
                    xn[i*(N+2)+j] = x(i,j) + inv_d*(b(i,j)-Ax_at(x,i,j));
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++) x(i,j) = xn[i*(N+2)+j];
            double rmax = residual_max(x, b);
            if (rmax > prev_r*1.01) dec = false;
            prev_r = rmax;
        }
        check(dec, "Jacobi residual decreases monotonically");
    }

    return test_summary();
}

// Unit tests for AMGPCG — exercises code in ../solvers/amgpcg.cpp
#define TEST_MODE
#include "../solvers/amgpcg.cpp"
#include "test_utils.h"

int main() {
    test_header("AMGPCG Unit Tests");

    // Test 1: amgpcg_solve returns accurate solution (sin problem)
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
                max_err = std::max(max_err,
                    std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 1e-2, "AMGPCG sin solution error < 1e-2");
        check(iters <= 30, "AMGPCG converges in <= 30 iterations");
    }

    // Test 2: AMGPCG far faster than unpreconditioned CG (polynomial RHS)
    {
        int N = 64;
        Grid b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double xi = i * b.h, yj = j * b.h;
                b(i,j) = 2.0 * (xi*(1-xi) + yj*(1-yj));
            }

        // Unpreconditioned CG
        Grid x_cg(N);
        Grid r = b, p = r;
        double rsold = dot(r, r);
        int cg_iters = 500;
        for (int k = 0; k < 500; k++) {
            Grid Ap = matvec(p);
            double pAp = dot(p, Ap);
            if (pAp < 1e-15) break;
            double alpha = rsold / pAp;
            axpy(alpha, p, x_cg);
            axpy(-alpha, Ap, r);
            double rsnew = dot(r, r);
            if (std::sqrt(rsnew) < 1e-8) { cg_iters = k + 1; break; }
            double beta = rsnew / rsold;
            for (int i = 1; i <= p.N; i++)
                for (int j = 1; j <= p.N; j++)
                    p(i,j) = r(i,j) + beta * p(i,j);
            rsold = rsnew;
        }

        int amg_iters = 0;
        Grid x_amg = amgpcg_solve(b, 30, 1e-8, amg_iters);

        check(amg_iters < cg_iters / 2,
            "AMGPCG < 1/2 CG its at N=64 (AMG=" + std::to_string(amg_iters) +
            ", CG=" + std::to_string(cg_iters) + ")");
    }

    // Test 3: Near grid-independent convergence
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
        check(std::abs(it64 - it32) <= 5,
            "iterations nearly independent of N (|it64-it32|=" +
            std::to_string(std::abs(it64-it32)) + ")");
    }

    // Test 4: Polynomial RHS solution
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
        check(max_err < 2e-3, "AMGPCG polynomial solution within discretization error");
    }

    return test_summary();
}

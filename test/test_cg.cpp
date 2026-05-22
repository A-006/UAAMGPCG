// Unit tests for CG solver — exercises code in ../solvers/cg.cpp
#define TEST_MODE
#include "../solvers/cg.cpp"
#include "test_utils.h"

int main() {
    test_header("Conjugate Gradient Unit Tests");

    // Test 1: matvec correctness — compare with direct Ax_at
    {
        int N = 8;
        Grid v(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                v(i,j) = std::sin(M_PI * i * v.h) * std::sin(M_PI * j * v.h);

        Grid Av = matvec(v);
        bool ok = true;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double expected = (4*v(i,j) - v(i+1,j) - v(i-1,j) - v(i,j+1) - v(i,j-1)) / (v.h*v.h);
                if (std::abs(Av(i,j) - expected) > 1e-12) ok = false;
            }
        check(ok, "matvec matches direct 5-point Laplacian");
    }

    // Test 2: dot product — internal points only
    {
        int N = 8;
        Grid a(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                a(i,j) = 1.0;
                b(i,j) = 2.0;
            }
        double s = dot(a, b);
        check(std::abs(s - 2.0*N*N) < 1e-12, "dot product correct sum");
    }

    // Test 3: CG solves sin problem accurately
    {
        int N = 32;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        // Run CG (inlined from solver main)
        Grid r = b;
        Grid p = r;
        double rsold = dot(r, r);
        int cg_iters = 0;
        for (int k = 0; k < 200; k++) {
            Grid Ap = matvec(p);
            double alpha = rsold / dot(p, Ap);
            axpy(alpha, p, x);
            axpy(-alpha, Ap, r);
            double rsnew = dot(r, r);
            if (std::sqrt(rsnew) < 1e-8) { cg_iters = k + 1; break; }
            double beta = rsnew / rsold;
            for (int i = 1; i <= p.N; i++)
                for (int j = 1; j <= p.N; j++)
                    p(i,j) = r(i,j) + beta * p(i,j);
            rsold = rsnew;
        }

        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err,
                    std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 2e-3, "CG sin solution error within discretization error at N=32");
        check(cg_iters < 100, "CG converges in < 100 iters at N=32");
    }

    // Test 4: CG much faster than Jacobi
    {
        int N = 64;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }
        Grid r = b, p = r;
        double rsold = dot(r, r);
        int iters = 0;
        for (int k = 0; k < 300; k++) {
            Grid Ap = matvec(p);
            axpy(rsold / dot(p, Ap), p, x);
            axpy(-rsold / dot(p, Ap), Ap, r);
            double rsnew = dot(r, r);
            if (std::sqrt(rsnew) < 1e-8) { iters = k + 1; break; }
            double beta = rsnew / rsold;
            for (int i = 1; i <= p.N; i++)
                for (int j = 1; j <= p.N; j++)
                    p(i,j) = r(i,j) + beta * p(i,j);
            rsold = rsnew;
        }
        check(iters < 150, "CG N=64 converges in < 150 iters (Jacobi ~8000)");
    }

    return test_summary();
}

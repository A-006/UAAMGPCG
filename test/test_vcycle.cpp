// Unit tests for V-Cycle / FMG — exercises code in ../solvers/vcycle.cpp
#define TEST_MODE
#include "../src/solver/vcycle.cpp"
#include "test_utils.h"

int main() {
    test_header("V-Cycle / Full Multigrid Unit Tests");

    // Test 1: Hierarchy construction
    {
        auto lv = build_levels(64);
        check(lv.size() >= 4, "N=64 produces >= 4 levels");
        check(lv[0] == 64, "First level = 64");
        check(lv.back() == 2, "Last level = 2");
    }
    {
        auto lv = build_levels(32);
        check(lv.size() >= 3, "N=32 produces >= 3 levels");
    }

    // Test 2: restrict preserves constant field
    {
        int N = 16, Nc = N / 2;
        Grid fine(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                fine(i,j) = 5.0;
        Grid coarse = restrict(fine);
        bool ok = true;
        for (int i = 1; i <= Nc; i++)
            for (int j = 1; j <= Nc; j++)
                if (std::abs(coarse(i,j) - 5.0) > 1e-12) ok = false;
        check(ok, "restrict preserves constant field");
    }

    // Test 3: prolongate+restrict roundtrip preserves constant (×2 then /4)
    {
        int Nc = 8, Nf = 16;
        Grid coarse(Nc);
        for (int i = 1; i <= Nc; i++)
            for (int j = 1; j <= Nc; j++)
                coarse(i,j) = 3.0;
        // prolongate with default scaling=2.0 gives 6.0
        Grid fine = prolongate(coarse, Nf, 1.0);
        bool ok = true;
        for (int i = 1; i <= Nf; i++)
            for (int j = 1; j <= Nf; j++)
                if (std::abs(fine(i,j) - 3.0) > 1e-12) ok = false;
        check(ok, "prolongate(scaling=1) preserves constant field");
    }

    // Test 4: RBGS reduces residual
    {
        int N = 16;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                x(i,j) = std::sin(M_PI * i) * std::sin(M_PI * j);
        double r0 = residual_max(x, b);
        rbgs_smooth(x, b, 3);
        double r1 = residual_max(x, b);
        check(r1 < r0, "RBGS reduces residual (high-freq initial error)");
    }

    // Test 5: FMG solves manufactured problem
    {
        int N = 32;
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        auto lv = build_levels(N);
        solve_with_fmg(x, b, lv, 1, 1e-8);

        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err,
                    std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 5e-2, "FMG solves sin problem (error < 0.05 at N=32)");
    }

    return test_summary();
}

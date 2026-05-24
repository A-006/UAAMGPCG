#include "../test_common.h"
#include "../test_utils.h"
#include <algorithm>

// Recursive V-Cycle (from solver/vcycle.cpp)
static void v_cycle(std::vector<Grid*>& x_lv, std::vector<Grid*>& b_lv, int level) {
    Grid& x = *x_lv[level];
    const Grid& b = *b_lv[level];
    int nl = (int)x_lv.size();
    if (level == nl - 1) { rbgs_smooth(x, b, 20); return; }

    rbgs_smooth(x, b, 2);
    Grid r(x.N);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++) r(i,j) = b(i,j) - Ax_at(x, i, j);
    Grid r_c = restrict(r);
    Grid& b_c = *b_lv[level+1];
    for (int i = 1; i <= b_c.N; i++)
        for (int j = 1; j <= b_c.N; j++) b_c(i,j) = r_c(i,j);
    Grid& x_c = *x_lv[level+1];
    std::fill(x_c.v.begin(), x_c.v.end(), 0.0);
    v_cycle(x_lv, b_lv, level+1);
    Grid corr = prolongate(x_c, x.N, 1.0);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++) x(i,j) += corr(i,j);
    rbgs_smooth(x, b, 2);
}

int main() {
    test_header("V-Cycle / Full Multigrid Unit Tests");

    // Test 1: Hierarchy
    {
        auto lv = build_levels(64);
        check(lv.size() >= 4 && lv[0] == 64 && lv.back() == 2, "N=64: 64→...→2");
    }
    { check(build_levels(32).size() >= 3, "N=32: >= 3 levels"); }

    // Test 2: restrict preserves constant
    {
        Grid fine(16);
        for (int i = 1; i <= 16; i++)
            for (int j = 1; j <= 16; j++) fine(i,j) = 5.0;
        Grid coarse = restrict(fine);
        bool ok = true;
        for (int i = 1; i <= 8; i++)
            for (int j = 1; j <= 8; j++)
                if (std::abs(coarse(i,j) - 5.0) > 1e-12) ok = false;
        check(ok, "restrict preserves constant field");
    }

    // Test 3: prolongate(scaling=1) preserves constant
    {
        Grid coarse(8);
        for (int i = 1; i <= 8; i++)
            for (int j = 1; j <= 8; j++) coarse(i,j) = 3.0;
        Grid fine = prolongate(coarse, 16, 1.0);
        bool ok = true;
        for (int i = 1; i <= 16; i++)
            for (int j = 1; j <= 16; j++)
                if (std::abs(fine(i,j) - 3.0) > 1e-12) ok = false;
        check(ok, "prolongate(scaling=1) preserves constant");
    }

    // Test 4: RBGS reduces residual
    {
        Grid x(16), b(16);
        for (int i = 1; i <= 16; i++)
            for (int j = 1; j <= 16; j++)
                x(i,j) = std::sin(M_PI * i) * std::sin(M_PI * j);
        double r0 = residual_max(x, b);
        rbgs_smooth(x, b, 3);
        check(residual_max(x, b) < r0, "RBGS reduces high-frequency residual");
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
        int nl = (int)lv.size();
        std::vector<Grid> s_x, s_b;
        for (int n : lv) { s_x.emplace_back(n); s_b.emplace_back(n); }
        std::vector<Grid*> xp, bp;
        for (int i = 0; i < nl; i++) { xp.push_back(&s_x[i]); bp.push_back(&s_b[i]); }

        // FMG
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) s_b[0](i,j) = b(i,j);
        for (int l = 1; l < nl; l++) s_b[l] = restrict(s_b[l-1]);
        rbgs_smooth(s_x[nl-1], s_b[nl-1], 50);
        for (int l = nl-2; l >= 0; l--) {
            Grid t = prolongate(s_x[l+1], lv[l], 1.0);
            for (int i = 1; i <= lv[l]; i++)
                for (int j = 1; j <= lv[l]; j++) s_x[l](i,j) = t(i,j);
            // Re-restrict b from fine grid for this V-Cycle
            s_b[0] = b;
            for (int k = 1; k < nl; k++) s_b[k] = restrict(s_b[k-1]);
            v_cycle(xp, bp, l);
        }

        double max_err = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                max_err = std::max(max_err, std::abs(s_x[0](i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));
        check(max_err < 5e-2, "FMG solves sin problem (error < 0.05)");
    }

    return test_summary();
}

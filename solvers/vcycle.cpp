// ===========================================================================
// Step 4: V-Cycle Multigrid — 2D Matrix-Free Multigrid
// Compile: g++ -std=c++17 -O2 vcycle.cpp -o vcycle
// Run: ./vcycle [N] [max_iter]
// ===========================================================================
//
// [Core Idea of Multigrid]
// Jacobi/RBGS eliminates high-frequency errors quickly, but low-frequency errors barely move.
// Low-frequency errors "look like high-frequency" on coarse grids → smoothing on coarse grids is also effective!
//
// V-Cycle Flow (Paper Algorithm 3):
//   Fine level: pre-smoothing (RBGS) → compute residual
//   Restriction: residual 4→1 averaged to coarse level
//   Coarse level: recursively do the same
//   Coarsest level: multiple RBGS accurate solve
//   Prolongation: coarse solution 1→4 interpolated back to fine level
//   Fine level: correct solution + post-smoothing (RBGS)
//
// Effect: regardless of grid size, convergence speed is nearly constant (grid-size independent!)
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

// ── Grid ──
struct Grid {
    int N;
    double h;
    std::vector<double> v;
    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}
    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

// ── Matrix-Free Ax — 5-point Laplacian ──
double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

double residual_max(const Grid& x, const Grid& b) {
    double rmax = 0;
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            rmax = std::max(rmax, std::abs(b(i,j) - Ax_at(x, i, j)));
    return rmax;
}

// ── RBGS Smoothing (in-place, iterations times) ──
void rbgs_smooth(Grid& x, const Grid& b, int iterations = 1) {
    double inv_diag = x.h * x.h / 4.0;
    int N = x.N;
    for (int sweep = 0; sweep < iterations; sweep++) {
        // red points
        for (int i = 1; i <= N; i++)
            for (int j = 1 + (i % 2); j <= N; j += 2)
                x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
        // black points
        for (int i = 1; i <= N; i++)
            for (int j = 1 + ((i+1) % 2); j <= N; j += 2)
                x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
    }
}

// ── Restriction: fine(2N) → coarse(N), average each 2×2 block (4→1) ──
Grid restrict(const Grid& r_fine) {
    int Nc = r_fine.N / 2;
    if (Nc < 2) Nc = 2;
    if (Nc % 2 != 0 && Nc > 2) Nc--; // ensure even for next coarsening
    Grid r_coarse(Nc);
    for (int ic = 1; ic <= Nc; ic++) {
        for (int jc = 1; jc <= Nc; jc++) {
            int i_f = 2*ic - 1;
            int j_f = 2*jc - 1;
            r_coarse(ic, jc) = 0.25 * (
                r_fine(i_f, j_f)     + r_fine(i_f+1, j_f) +
                r_fine(i_f, j_f+1)   + r_fine(i_f+1, j_f+1));
        }
    }
    return r_coarse;
}

// ── Prolongation: coarse(N) → fine(2N), constant interpolation × 2 ──
Grid prolongate(const Grid& x_coarse, int Nf, double scaling = 2.0) {
    Grid x_fine(Nf);
    int Nc = x_coarse.N;
    for (int ic = 1; ic <= Nc; ic++) {
        for (int jc = 1; jc <= Nc; jc++) {
            double val = x_coarse(ic, jc) * scaling;
            int i_f = 2*ic - 1;
            int j_f = 2*jc - 1;
            x_fine(i_f,   j_f)   = val;
            x_fine(i_f+1, j_f)   = val;
            x_fine(i_f,   j_f+1) = val;
            x_fine(i_f+1, j_f+1) = val;
        }
    }
    return x_fine;
}

// ── Build level list [N, N/2, N/4, ...] until minimum ≥ 2 ──
std::vector<int> build_levels(int N) {
    std::vector<int> levels;
    int n = N;
    while (n >= 2) {
        levels.push_back(n);
        if (n <= 3) break;
        n = n / 2;
        if (n < 2) n = 2;
    }
    return levels;
}

// ── Recursive V-Cycle ──
void v_cycle(const std::vector<Grid*>& x_levels,
             const std::vector<Grid*>& b_levels,
             int level, int /*max_level*/)
{
    Grid& x = *x_levels[level];
    const Grid& b = *b_levels[level];

    // Base case: multiple smoothings on the coarsest level
    if (level == (int)x_levels.size() - 1) {
        rbgs_smooth(x, b, 20);
        return;
    }

    // ── Downward stroke ──
    rbgs_smooth(x, b, 2);

    // Residual r = b - Ax
    Grid r(x.N);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            r(i,j) = b(i,j) - Ax_at(x, i, j);

    // Restrict to coarse level
    Grid r_coarse = restrict(r);
    Grid& b_coarse = *b_levels[level + 1];
    for (int i = 1; i <= b_coarse.N; i++)
        for (int j = 1; j <= b_coarse.N; j++)
            b_coarse(i,j) = r_coarse(i,j);

    Grid& x_coarse = *x_levels[level + 1];
    std::fill(x_coarse.v.begin(), x_coarse.v.end(), 0.0);

    // Recursive call
    v_cycle(x_levels, b_levels, level + 1, x_levels.size() - 1);

    // ── Upward stroke ──
    Grid correction = prolongate(x_coarse, x.N, 1.0);  // no extra scaling for 2D
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            x(i,j) += correction(i,j);

    rbgs_smooth(x, b, 2);
}

// ── FMG (Full Multigrid) as standalone solver ──
// Start solving from the coarsest level, prolongate level by level to the fine level, one V-Cycle per level
void solve_with_fmg(Grid& x, const Grid& b, const std::vector<int>& levels,
                    int max_cycles, double tol) {

    int nl = (int)levels.size();

    // Pre-allocate x and b storage for all levels
    std::vector<Grid> storage_x, storage_b;
    for (int n : levels) {
        storage_x.emplace_back(n);
        storage_b.emplace_back(n);
    }

    std::vector<Grid*> x_ptrs, b_ptrs;
    for (int lv = 0; lv < nl; lv++) {
        x_ptrs.push_back(&storage_x[lv]);
        b_ptrs.push_back(&storage_b[lv]);
    }

    // Coarsest level: restrict b down, solve directly
    // Restrict b level by level
    Grid b_fine(x.N);
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            b_fine(i,j) = b(i,j);
    storage_b[0] = b_fine;

    for (int lv = 1; lv < nl; lv++)
        storage_b[lv] = restrict(storage_b[lv-1]);

    // Solve directly on the coarsest level
    rbgs_smooth(storage_x[nl-1], storage_b[nl-1], 50);

    // Ascend level by level, one V-Cycle per level
    for (int lv = nl - 2; lv >= 0; lv--) {
        // Prolongate coarse solution to current level
        Grid temp = prolongate(storage_x[lv+1], levels[lv], 1.0);
        for (int i = 1; i <= levels[lv]; i++)
            for (int j = 1; j <= levels[lv]; j++)
                storage_x[lv](i,j) = temp(i,j);

        // Perform several V-Cycles
        for (int cyc = 0; cyc < max_cycles; cyc++) {
            v_cycle(x_ptrs, b_ptrs, lv, nl-1);
            double rmax = residual_max(storage_x[lv], storage_b[lv]);
            if (rmax < tol) break;
        }
    }

    // Copy the finest level result
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            x(i,j) = storage_x[0](i,j);
}

int main(int argc, char* argv[]) {
    int N = 64;
    int max_cycles = 100;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) max_cycles = std::atoi(argv[2]);

    const double TOL = 1e-8;

    std::cout << "=== V-Cycle Multigrid  N=" << N
              << " (" << N*N << " unknowns) ===\n\n";

    Grid x(N), b(N);

    // Known solution u = sin(pi*x)sin(pi*y)
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) {
            double sx = std::sin(M_PI * i * x.h);
            double sy = std::sin(M_PI * j * x.h);
            b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
        }

    auto levels = build_levels(N);
    std::cout << "Levels: ";
    for (int n : levels) std::cout << n << " ";
    std::cout << "\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    solve_with_fmg(x, b, levels, 1, TOL);  // 1 V-Cycle per level in FMG
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double err = 0;
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            err = std::max(err, std::abs(x(i,j) - std::sin(M_PI*i*x.h)*std::sin(M_PI*j*x.h)));

    std::cout << "\nTime: " << elapsed << " s, Error: " << err << '\n';
    std::cout << "\nKey feature of FMG: convergence speed hardly deteriorates as N increases\n";
    return 0;
}

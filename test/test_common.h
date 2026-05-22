#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// Shared Grid for standalone solver tests (uniform grid, Dirichlet BC)
struct Grid {
    int N;
    double h;
    std::vector<double> v;
    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}
    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

inline double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

inline double dot(const Grid& a, const Grid& b) {
    double s = 0;
    for (int i = 1; i <= a.N; i++)
        for (int j = 1; j <= a.N; j++)
            s += a(i,j) * b(i,j);
    return s;
}

inline void axpy(double a, const Grid& x, Grid& y) {
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            y(i,j) += a * x(i,j);
}

inline Grid matvec(const Grid& v) {
    Grid Av(v.N);
    for (int i = 1; i <= v.N; i++)
        for (int j = 1; j <= v.N; j++)
            Av(i,j) = Ax_at(v, i, j);
    return Av;
}

inline double residual_max(const Grid& x, const Grid& b) {
    double rmax = 0;
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            rmax = std::max(rmax, std::abs(b(i,j) - Ax_at(x, i, j)));
    return rmax;
}

// ── RBGS ──
inline void rbgs_smooth(Grid& x, const Grid& b, int iterations = 1) {
    double inv_diag = x.h * x.h / 4.0;
    int N = x.N;
    for (int sweep = 0; sweep < iterations; sweep++) {
        for (int i = 1; i <= N; i++)
            for (int j = 1 + (i % 2); j <= N; j += 2)
                x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
        for (int i = 1; i <= N; i++)
            for (int j = 1 + ((i+1) % 2); j <= N; j += 2)
                x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
    }
}

// ── Multigrid ──
inline std::vector<int> build_levels(int N) {
    std::vector<int> lv;
    int n = N;
    while (n >= 2) { lv.push_back(n); if (n <= 3) break; n = n/2; if (n < 2) n = 2; }
    return lv;
}

inline Grid restrict(const Grid& r_fine) {
    int Nc = r_fine.N / 2;
    if (Nc < 2) Nc = 2;
    if (Nc % 2 != 0 && Nc > 2) Nc--;
    Grid r_coarse(Nc);
    for (int ic = 1; ic <= Nc; ic++)
        for (int jc = 1; jc <= Nc; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            r_coarse(ic, jc) = 0.25 * (r_fine(i_f,j_f) + r_fine(i_f+1,j_f)
                                     + r_fine(i_f,j_f+1) + r_fine(i_f+1,j_f+1));
        }
    return r_coarse;
}

inline Grid prolongate(const Grid& x_coarse, int Nf, double scaling = 2.0) {
    Grid x_fine(Nf);
    int Nc = x_coarse.N;
    for (int ic = 1; ic <= Nc; ic++)
        for (int jc = 1; jc <= Nc; jc++) {
            double val = x_coarse(ic, jc) * scaling;
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            x_fine(i_f,j_f) = x_fine(i_f+1,j_f) = x_fine(i_f,j_f+1) = x_fine(i_f+1,j_f+1) = val;
        }
    return x_fine;
}

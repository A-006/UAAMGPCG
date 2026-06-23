/**
 * @file pcg.cpp
 * @brief Preconditioned Conjugate Gradient — CG accelerated by a pluggable preconditioner.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/pcg.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
// ── One-shot export of the EXACT linear system PCG solves, for external
// solver tests (e.g. NVIDIA AMGX). A = (-nabla^2) with the same Neumann/solid
// handling as PCG::solve's matvec; b = -(rhs - mean) (zero-mean, consistent).
// MatrixMarket coordinate (A) + array (b). Unknowns = non-solid interior cells,
// numbered 1-based in column-major (i fastest) order.
void dump_linear_system(const Grid& g, const std::vector<double>& rhs_in, const std::string& prefix) {
    const int nx = g.nx, ny = g.ny;

    std::vector<int> id(g.p_size(), 0);
    int n = 0;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            if (!g.is_solid(i, j))
                id[g.ip(i, j)] = ++n;

    const bool var      = g.has_variable_lap();
    const double idx2   = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy);
    const double diag_c = 2.0 * (idx2 + idy2);

    std::vector<int> rows, cols;
    std::vector<double> vals;
    auto emit = [&](int r, int c, double v) {
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
    };

    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            if (g.is_solid(i, j))
                continue;
            const int r     = id[g.ip(i, j)];
            const int idx   = g.ip(i, j);
            const double cx = var ? g.lap_off_x[idx] : -idx2;
            const double cy = var ? g.lap_off_y[idx] : -idy2;
            double d        = var ? g.lap_diag[idx] : diag_c;

            // L, R (x-coeff cx); B, T (y-coeff cy). In matvec a missing/solid
            // neighbor uses v[idx] itself, i.e. folds its coeff into the diagonal.
            const int ni[4][2]   = {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};
            const double coeff[4] = {cx, cx, cy, cy};
            for (int k = 0; k < 4; ++k) {
                int a = ni[k][0], b = ni[k][1];
                bool inrange = (a >= 1 && a <= nx && b >= 1 && b <= ny);
                if (inrange && !g.is_solid(a, b))
                    emit(r, id[g.ip(a, b)], coeff[k]);
                else
                    d += coeff[k];
            }
            emit(r, r, d);
        }

    std::ofstream fa(prefix + "_A.mtx");
    fa << "%%MatrixMarket matrix coordinate real general\n";
    fa << n << " " << n << " " << rows.size() << "\n";
    fa.precision(17);
    for (size_t t = 0; t < rows.size(); ++t)
        fa << rows[t] << " " << cols[t] << " " << vals[t] << "\n";

    // RHS exactly as PCG forms it: zero-mean over non-solid, then negated.
    double sum = 0;
    int cnt    = 0;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            if (!g.is_solid(i, j)) {
                sum += rhs_in[g.ip(i, j)];
                ++cnt;
            }
    const double mean = (cnt > 0) ? sum / cnt : 0.0;

    std::ofstream fb(prefix + "_b.mtx");
    fb << "%%MatrixMarket matrix array real general\n";
    fb << n << " 1\n";
    fb.precision(17);
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            if (!g.is_solid(i, j))
                fb << -(rhs_in[g.ip(i, j)] - mean) << "\n";
}

// Dump UAAMGPCG's own converged solution g.p over the same non-solid cells, in
// the same 1-based column-major order as the exported A/b. Written at solve
// exit so the values are the final, converged pressure.
void dump_solution(const Grid& g, const std::string& prefix) {
    std::ofstream fx(prefix + "_x_uaamg.txt");
    fx.precision(17);
    for (int j = 1; j <= g.ny; ++j)
        for (int i = 1; i <= g.nx; ++i)
            if (!g.is_solid(i, j))
                fx << g.p[g.ip(i, j)] << "\n";
}
} // namespace

PCG::PCG(std::unique_ptr<Preconditioner> p) : precond_(std::move(p)) {}

std::string PCG::name() const {
    return "PCG(" + precond_->name() + ")";
}

void PCG::solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol) {
    const int nx = g.nx, ny = g.ny;

    // One-shot export of the actual system (env UAAMG_DUMP_MTX=<prefix>), then
    // continue solving normally. Used to feed a real CFD matrix to AMGX etc.
    // The guard dumps g.p on function exit (any return) so the saved solution
    // is the converged one, for cross-checking against AMGX on the same system.
    struct SolDumpGuard {
        bool armed;
        const Grid& g;
        std::string prefix;
        ~SolDumpGuard() {
            if (armed)
                dump_solution(g, prefix);
        }
    };
    SolDumpGuard guard{false, g, ""};
    if (const char* dump_prefix = std::getenv("UAAMG_DUMP_MTX")) {
        static bool dumped = false;
        if (!dumped) {
            dumped      = true;
            guard.armed = true;
            guard.prefix = dump_prefix;
            dump_linear_system(g, rhs_in, dump_prefix);
        }
    }

    // Zero-mean RHS, then negate because matvec is -nabla^2
    // PCG solves (-nabla^2) p = -rhs_in  =>  nabla^2 p = rhs_in
    std::vector<double> rhs = rhs_in;
    {
        double sum = 0;
        int count  = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j)) {
                    sum += rhs[g.ip(i, j)];
                    count++;
                }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j))
                    rhs[g.ip(i, j)] = -(rhs[g.ip(i, j)] - mean);
    }

    auto matvec = [&](const std::vector<double>& v, std::vector<double>& Av) {
        if (g.has_variable_lap()) {
            for (int i = 1; i <= nx; i++)
                for (int j = 1; j <= ny; j++) {
                    int idx = g.ip(i, j);
                    if (g.is_solid(i, j)) {
                        Av[idx] = 0.0;
                        continue;
                    }
                    double vL = (i > 1 && !g.is_solid(i - 1, j)) ? v[g.ip(i - 1, j)] : v[idx];
                    double vR = (i < nx && !g.is_solid(i + 1, j)) ? v[g.ip(i + 1, j)] : v[idx];
                    double vB = (j > 1 && !g.is_solid(i, j - 1)) ? v[g.ip(i, j - 1)] : v[idx];
                    double vT = (j < ny && !g.is_solid(i, j + 1)) ? v[g.ip(i, j + 1)] : v[idx];
                    Av[idx]   = g.lap_diag[idx] * v[idx] + g.lap_off_x[idx] * (vL + vR) +
                                g.lap_off_y[idx] * (vB + vT);
                }
        } else {
            double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy);
            double diag = 2.0 * (idx2 + idy2);
            for (int i = 1; i <= nx; i++)
                for (int j = 1; j <= ny; j++) {
                    int idx = g.ip(i, j);
                    if (g.is_solid(i, j)) {
                        Av[idx] = 0.0;
                        continue;
                    }
                    double vL = (i > 1 && !g.is_solid(i - 1, j)) ? v[g.ip(i - 1, j)] : v[idx];
                    double vR = (i < nx && !g.is_solid(i + 1, j)) ? v[g.ip(i + 1, j)] : v[idx];
                    double vB = (j > 1 && !g.is_solid(i, j - 1)) ? v[g.ip(i, j - 1)] : v[idx];
                    double vT = (j < ny && !g.is_solid(i, j + 1)) ? v[g.ip(i, j + 1)] : v[idx];
                    Av[idx]   = diag * v[idx] - (vL + vR) * idx2 - (vB + vT) * idy2;
                }
        }
    };

    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j))
                    s += a[g.ip(i, j)] * b[g.ip(i, j)];
        return s;
    };

    auto subtract_mean = [&](std::vector<double>& v) {
        double sum = 0;
        int count  = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j)) {
                    sum += v[g.ip(i, j)];
                    count++;
                }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j))
                    v[g.ip(i, j)] -= mean;
    };

    // r = b - A*x  (x=0 initially)
    std::vector<double> r = rhs;
    subtract_mean(r);

    // z = M^{-1} * r
    std::vector<double> z(rhs.size());
    precond_->apply(g, r, z);
    subtract_mean(z);

    std::vector<double> p = z;
    double rsold          = dot(r, z);

    std::vector<double> Ap(rhs.size());

    for (int k = 0; k < max_iter; k++) {
        matvec(p, Ap);

        double pAp = dot(p, Ap);
        if (pAp < 1e-15)
            return;

        double alpha = rsold / pAp;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j)) {
                    int idx = g.ip(i, j);
                    g.p[idx] += alpha * p[idx];
                    r[idx] -= alpha * Ap[idx];
                }

        double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < tol)
            return;

        precond_->apply(g, r, z);
        subtract_mean(z);

        double beta = dot(r, z) / rsold;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i, j)) {
                    int idx = g.ip(i, j);
                    p[idx]  = z[idx] + beta * p[idx];
                }
        rsold = dot(r, z);
    }
}

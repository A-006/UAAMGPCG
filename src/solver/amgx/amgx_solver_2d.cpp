/**
 * @file amgx_solver.cpp
 * @brief NVIDIA AMGX 2D Poisson backend (see amgx_solver.h).
 * @author liutao
 * @date 2026-06-23
 */
#include "solver/amgx/amgx_solver_2d.h"
#include "mesh/grid_2d.h"
#include "solver/amgx/amgx_backend.h"

#include <algorithm>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// CSR assembly of A = -nabla^2 over the non-solid interior cells, using exactly
// the 5-point stencil PCG::solve applies in its matvec (a missing/solid
// neighbour folds its coefficient onto the diagonal => zero-flux Neumann).
// 0-based CSR, columns sorted per row. Also produces the zero-mean, negated RHS
// the iterative solvers use, and the flat grid index of each unknown.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

struct LinearSystem {
    int n = 0;
    std::vector<int> row_ptr, col_idx, cell_of;
    std::vector<double> vals, b;
};

LinearSystem assemble(const Grid& g, const std::vector<double>& rhs_in) {
    const int nx = g.nx, ny = g.ny;

    std::vector<int> id(g.p_size(), -1);
    LinearSystem s;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            if (!g.is_solid(i, j)) {
                id[g.ip(i, j)] = s.n++;
                s.cell_of.push_back(g.ip(i, j));
            }

    const bool var      = g.has_variable_lap();
    const double idx2   = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy);
    const double diag_c = 2.0 * (idx2 + idy2);

    s.row_ptr.reserve(s.n + 1);
    s.row_ptr.push_back(0);
    std::vector<std::pair<int, double>> row;

    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            if (g.is_solid(i, j))
                continue;
            const int idx   = g.ip(i, j);
            const double cx = var ? g.lap_off_x[idx] : -idx2;
            const double cy = var ? g.lap_off_y[idx] : -idy2;
            double d        = var ? g.lap_diag[idx] : diag_c;

            const int ni[4][2]    = {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};
            const double coeff[4] = {cx, cx, cy, cy};
            row.clear();
            for (int k = 0; k < 4; ++k) {
                int a = ni[k][0], b = ni[k][1];
                bool inrange = (a >= 1 && a <= nx && b >= 1 && b <= ny);
                if (inrange && !g.is_solid(a, b))
                    row.emplace_back(id[g.ip(a, b)], coeff[k]);
                else
                    d += coeff[k];
            }
            row.emplace_back(id[idx], d);
            std::sort(row.begin(), row.end());
            for (auto& e : row) {
                s.col_idx.push_back(e.first);
                s.vals.push_back(e.second);
            }
            s.row_ptr.push_back(static_cast<int>(s.col_idx.size()));
        }

    double sum = 0;
    for (int u = 0; u < s.n; ++u)
        sum += rhs_in[s.cell_of[u]];
    const double mean = (s.n > 0) ? sum / s.n : 0.0;
    s.b.resize(s.n);
    for (int u = 0; u < s.n; ++u)
        s.b[u] = -(rhs_in[s.cell_of[u]] - mean);

    return s;
}

} // namespace

struct AMGXSolver::Impl {
    AmgxBackend backend; // throws here (via factory) if built without AMGX
};

AMGXSolver::AMGXSolver() : impl_(std::make_unique<Impl>()) {}
AMGXSolver::~AMGXSolver() = default;

std::string AMGXSolver::name() const {
    return "AMGX";
}

void AMGXSolver::solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol) {
    LinearSystem s = assemble(g, rhs_in);
    if (s.n == 0)
        return;

    std::vector<double> xh =
        impl_->backend.solve_csr(s.n, s.row_ptr, s.col_idx, s.vals, s.b, max_iter, tol);

    // Match the project's zero-mean pressure convention (the system is singular
    // up to a constant); pin solid cells to 0.
    double sum = 0;
    for (double v : xh)
        sum += v;
    const double mean = xh.empty() ? 0.0 : sum / xh.size();
    std::fill(g.p.begin(), g.p.end(), 0.0);
    for (int u = 0; u < s.n; ++u)
        g.p[s.cell_of[u]] = xh[u] - mean;
}

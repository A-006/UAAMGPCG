/**
 * @file amgx_solver_3d.cpp
 * @brief NVIDIA AMGX 3D Poisson backend (see amgx_solver_3d.h).
 * @author liutao
 * @date 2026-06-23
 */
#include "solver/amgx_solver_3d.h"
#include "mesh/grid_3d.h"
#include "solver/amgx_backend.h"

#include <algorithm>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// CSR assembly of A = -nabla^2 over the non-solid interior cells, using exactly
// the 7-point stencil PCG3D::solve applies in its matvec (a missing/solid
// neighbour folds its coefficient onto the diagonal => zero-flux Neumann).
// Grid3D is constant-coefficient. 0-based CSR, columns sorted per row. Also
// produces the zero-mean, negated RHS and each unknown's flat grid index.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

struct LinearSystem {
    int n = 0;
    std::vector<int> row_ptr, col_idx, cell_of;
    std::vector<double> vals, b;
};

LinearSystem assemble(const Grid3D& g, const std::vector<double>& rhs_in) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;

    std::vector<int> id(g.p_size(), -1);
    LinearSystem s;
    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            for (int k = 1; k <= nz; ++k)
                if (!g.is_solid(i, j, k)) {
                    id[g.ip(i, j, k)] = s.n++;
                    s.cell_of.push_back(g.ip(i, j, k));
                }

    const double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy), idz2 = 1.0 / (g.dz * g.dz);
    const double diag = 2.0 * (idx2 + idy2 + idz2);

    s.row_ptr.reserve(s.n + 1);
    s.row_ptr.push_back(0);
    std::vector<std::pair<int, double>> row;

    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            for (int k = 1; k <= nz; ++k) {
                if (g.is_solid(i, j, k))
                    continue;
                double d = diag;
                const int ni[6][3] = {{i - 1, j, k}, {i + 1, j, k}, {i, j - 1, k},
                                      {i, j + 1, k}, {i, j, k - 1}, {i, j, k + 1}};
                const double coeff[6] = {-idx2, -idx2, -idy2, -idy2, -idz2, -idz2};
                row.clear();
                for (int t = 0; t < 6; ++t) {
                    int a = ni[t][0], b = ni[t][1], c = ni[t][2];
                    bool inrange = (a >= 1 && a <= nx && b >= 1 && b <= ny && c >= 1 && c <= nz);
                    if (inrange && !g.is_solid(a, b, c))
                        row.emplace_back(id[g.ip(a, b, c)], coeff[t]);
                    else
                        d += coeff[t];
                }
                row.emplace_back(id[g.ip(i, j, k)], d);
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

struct AMGXSolver3D::Impl {
    AmgxBackend backend; // throws here (via factory) if built without AMGX
};

AMGXSolver3D::AMGXSolver3D() : impl_(std::make_unique<Impl>()) {}
AMGXSolver3D::~AMGXSolver3D() = default;

std::string AMGXSolver3D::name() const {
    return "AMGX3D";
}

void AMGXSolver3D::solve(Grid3D& g, const std::vector<double>& rhs_in, int max_iter, double tol) {
    LinearSystem s = assemble(g, rhs_in);
    if (s.n == 0)
        return;

    std::vector<double> xh =
        impl_->backend.solve_csr(s.n, s.row_ptr, s.col_idx, s.vals, s.b, max_iter, tol);

    double sum = 0;
    for (double v : xh)
        sum += v;
    const double mean = xh.empty() ? 0.0 : sum / xh.size();
    std::fill(g.p.begin(), g.p.end(), 0.0);
    for (int u = 0; u < s.n; ++u)
        g.p[s.cell_of[u]] = xh[u] - mean;
}

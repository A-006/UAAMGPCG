// ─────────────────────────────────────────────────────────────────────────
// Composite finite-volume Poisson operator (3D). Mirror of the 2D operator with
// PHYSICAL h-dependent coefficients (in 3D κ_same = face_area/h = h²/h = h):
//   * same-level face : κ_same = h_l      (diag += h_l, off = -h_l)
//   * coarse/fine face: rank-1 PSD stiffness K = κ_face·wwᵀ, κ_face = (4/3)h_l,
//     w_coarse = 1, w_fine = -1/4 over the m=4 fine children → symmetric,
//     conservative, consistent (centred transversely-averaged gradient).
//   * Dirichlet boundary face: κ_dir = 2 h_l (ghost at the face, distance h/2).
// A x ≈ V_C·(-∇²p)_C with V_C = h_l³; numerical Laplacian = (A x)/V.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "adaptive_grid_3d.h"
#include "sparse.h"
#include <vector>
#include <functional>

namespace semistruct {

struct PoissonOperator3D {
    const AdaptiveGrid3D* g = nullptr;
    CSR A;
    std::vector<double> vol;  // per-DOF cell volume h_l³
    std::vector<std::vector<AdaptiveGrid3D::DirFace>> dir;

    void build(const AdaptiveGrid3D& grid) {
        g = &grid;
        int n = grid.ndof;
        vol.assign(n, 0.0);
        dir.assign(n, {});
        std::vector<std::map<int, double>> rows(n);
        for (int d = 0; d < n; ++d) {
            int l = grid.dof_level[d], i = grid.dof_i[d], j = grid.dof_j[d], k = grid.dof_k[d];
            double h = grid.lev[l].h;
            vol[d] = h * h * h;
            for (int s = 0; s < 6; ++s) {
                int ni = i + AdaptiveGrid3D::FDI[s], nj = j + AdaptiveGrid3D::FDJ[s],
                    nk = k + AdaptiveGrid3D::FDK[s];
                if (!grid.lev[l].in(ni, nj, nk)) {              // domain boundary
                    if (grid.bc[s] == BC::Dirichlet) {
                        double kappa = h / (0.5 * h) * h;        // = 2h (face area h², dist h/2 → h²/(h/2)=2h)
                        rows[d][d] += kappa;
                        dir[d].push_back({kappa, s});
                    }
                    continue;                                   // Neumann: no flux
                }
                Cell nc = grid.lev[l].at(ni, nj, nk);
                if (nc == Cell::LEAF) {
                    if (s == 0 || s == 2 || s == 4) {           // negative dirs: stamp once
                        int nd = grid.dofAt(l, ni, nj, nk);
                        stampSameLevel(rows, d, nd, h);
                    }
                } else if (nc == Cell::REFINED) {               // coarse/fine: from coarse side
                    stampCoarseFine(grid, rows, d, ni, nj, nk, s, h);
                }
                // OUTSIDE → fine side, handled from the coarse side
            }
        }
        std::vector<std::vector<std::pair<int, double>>> off(n);
        std::vector<double> diagv(n, 0.0);
        for (int d = 0; d < n; ++d)
            for (auto& e : rows[d]) {
                if (e.first == d) diagv[d] = e.second;
                else off[d].push_back(e);
            }
        A = buildCSR(n, off, diagv);
    }

private:
    static void stampSameLevel(std::vector<std::map<int, double>>& rows, int a, int b, double h) {
        rows[a][a] += h; rows[b][b] += h;
        rows[a][b] -= h; rows[b][a] -= h;
    }
    // 3D coarse/fine face: m=4 fine cells, κ_face = (4/3)h_coarse, w_fine = -1/4.
    static void stampCoarseFine(const AdaptiveGrid3D& grid, std::vector<std::map<int, double>>& rows,
                                int coarse, int ni, int nj, int nk, int s, double h) {
        int cs[4][3];
        AdaptiveGrid3D::childrenOnFace(ni, nj, nk, s, cs);
        int fl = grid.dof_level[coarse] + 1;
        int a[4];
        for (int t = 0; t < 4; ++t) a[t] = grid.dofAt(fl, cs[t][0], cs[t][1], cs[t][2]);
        const int m = 4;
        const double kf = (4.0 / 3.0) * h;
        double wC = 1.0, wF = -1.0 / m;
        rows[coarse][coarse] += kf * wC * wC;
        for (int t = 0; t < m; ++t) {
            rows[a[t]][a[t]] += kf * wF * wF;
            rows[coarse][a[t]] += kf * wC * wF;
            rows[a[t]][coarse] += kf * wC * wF;
            for (int u = t + 1; u < m; ++u) {
                rows[a[t]][a[u]] += kf * wF * wF;
                rows[a[u]][a[t]] += kf * wF * wF;
            }
        }
    }

public:
    std::vector<double> rhs(const std::function<double(double, double, double)>& f,
                            const std::function<double(double, double, double)>& gd) const {
        int n = g->ndof;
        std::vector<double> b(n, 0.0);
        for (int d = 0; d < n; ++d) {
            int l = g->dof_level[d], i = g->dof_i[d], j = g->dof_j[d], k = g->dof_k[d];
            double x = g->cx(l, i), y = g->cy(l, j), z = g->cz(l, k);
            b[d] = vol[d] * f(x, y, z);
            double hh = 0.5 * g->lev[l].h;
            for (auto& fc : dir[d]) {
                double gx = x, gy = y, gz = z;
                switch (fc.side) {
                    case 0: gx -= hh; break; case 1: gx += hh; break;
                    case 2: gy -= hh; break; case 3: gy += hh; break;
                    case 4: gz -= hh; break; default: gz += hh; break;
                }
                b[d] += fc.kappa * gd(gx, gy, gz);
            }
        }
        return b;
    }

    std::vector<double> sample(const std::function<double(double, double, double)>& f) const {
        int n = g->ndof;
        std::vector<double> v(n);
        for (int d = 0; d < n; ++d)
            v[d] = f(g->cx(g->dof_level[d], g->dof_i[d]), g->cy(g->dof_level[d], g->dof_j[d]),
                     g->cz(g->dof_level[d], g->dof_k[d]));
        return v;
    }

    double rmsV(const std::vector<double>& a, const std::vector<double>& b) const {
        double num = 0.0, den = 0.0;
        for (int d = 0; d < g->ndof; ++d) {
            double e = a[d] - b[d];
            num += vol[d] * e * e; den += vol[d];
        }
        return std::sqrt(num / den);
    }

    std::vector<double> numericalLaplacian(const std::vector<double>& p) const {
        std::vector<double> Ap;
        A.matvec(p, Ap);
        for (int d = 0; d < g->ndof; ++d) Ap[d] /= vol[d];
        return Ap;
    }
};

// Independent textbook 7-point Laplacian on a UNIFORM n³ grid (κ_same = h).
inline CSR uniformLaplacian7pt(int n, std::array<BC, 6> bc) {
    int N = n * n * n;
    double h = 1.0 / n;
    auto id = [&](int i, int j, int k) { return i + j * n + k * n * n; };
    std::vector<std::vector<std::pair<int, double>>> off(N);
    std::vector<double> diagv(N, 0.0);
    const int di[6] = {-1, 1, 0, 0, 0, 0};
    const int dj[6] = {0, 0, -1, 1, 0, 0};
    const int dk[6] = {0, 0, 0, 0, -1, 1};
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                int r = id(i, j, k);
                for (int s = 0; s < 6; ++s) {
                    int ni = i + di[s], nj = j + dj[s], nk = k + dk[s];
                    if (ni < 0 || ni >= n || nj < 0 || nj >= n || nk < 0 || nk >= n) {
                        if (bc[s] == BC::Dirichlet) diagv[r] += 2.0 * h;
                        continue;
                    }
                    diagv[r] += h;
                    off[r].push_back({id(ni, nj, nk), -h});
                }
            }
    return buildCSR(N, off, diagv);
}

}  // namespace semistruct

// ─────────────────────────────────────────────────────────────────────────
// Composite finite-volume Poisson operator on the semi-structured adaptive grid.
//
// Discretizes  ∫_C -∇²p dV = ∮_∂C -∇p·n dS  (eq. 2-3 of the paper) over leaf
// cells. Each face contributes a flux  kappa * (p_C - p_nb)  with the symmetric
// conservative coefficient kappa = subface_len / center_dist:
//   * same level     → kappa = 1
//   * coarse/fine    → kappa = 2/3 (sub-face h, centre distance 3h/2)  [eq.9-13]
// The coarse cell couples to the two fine cells touching the shared face, equal
// and opposite, so the operator is SYMMETRIC and flux-CONSERVATIVE across
// T-junctions (eq. 11: f1+f3 = f4). With at least one Dirichlet side it is SPD.
//
// (A x)_C approximates  V_C * (-∇²p)_C  with V_C = h_l² the leaf volume, so the
// numerical Laplacian is (A x)_C / V_C and the Poisson RHS is  b_C = V_C * f_C.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "adaptive_grid_2d.h"
#include "sparse.h"
#include <vector>
#include <functional>

namespace semistruct {

struct PoissonOperator2D {
    const AdaptiveGrid2D* g = nullptr;
    CSR A;
    std::vector<double> vol;  // per-DOF cell volume h_l²
    // Dirichlet bookkeeping: for each DOF, list of (kappa, side) faces on a
    // Dirichlet domain boundary (the ghost value gd enters the RHS).
    std::vector<std::vector<AdaptiveGrid2D::DirFace>> dir;

    void build(const AdaptiveGrid2D& grid) {
        g = &grid;
        int n = grid.ndof;
        vol.assign(n, 0.0);
        dir.assign(n, {});
        // Symmetric assembly into per-row maps (each off-diagonal kept symmetric).
        std::vector<std::map<int, double>> rows(n);
        for (int d = 0; d < n; ++d) {
            int l = grid.dof_level[d], i = grid.dof_i[d], j = grid.dof_j[d];
            double h = grid.lev[l].h;
            vol[d] = h * h;
            const int di[4] = {-1, 1, 0, 0};
            const int dj[4] = {0, 0, -1, 1};
            for (int s = 0; s < 4; ++s) {
                int ni = i + di[s], nj = j + dj[s];
                if (!grid.lev[l].in(ni, nj)) {                  // domain boundary
                    if (grid.bc[s] == BC::Dirichlet) {
                        double kappa = h / (0.5 * h);           // = 2 (ghost at face)
                        rows[d][d] += kappa;
                        dir[d].push_back({kappa, s});
                    }
                    continue;                                   // Neumann: no flux
                }
                Cell nc = grid.lev[l].at(ni, nj);
                if (nc == Cell::LEAF) {
                    if (s == 0 || s == 2) {                     // stamp same-level face once
                        int nd = grid.dofAt(l, ni, nj);
                        stampSameLevel(rows, d, nd);
                    }
                } else if (nc == Cell::REFINED) {               // coarse/fine: from coarse side
                    stampCoarseFine(grid, rows, d, ni, nj, s);
                }
                // OUTSIDE → fine side of a T-junction, handled from the coarse side
            }
        }
        // build CSR from symmetric rows (diagonal stored at [d][d])
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
    // Same-level face between leaves a,b: standard +1/-1 conservative stencil.
    static void stampSameLevel(std::vector<std::map<int, double>>& rows, int a, int b) {
        rows[a][a] += 1.0; rows[b][b] += 1.0;
        rows[a][b] -= 1.0; rows[b][a] -= 1.0;
    }
    // Coarse/fine face: consistent, symmetric, conservative rank-1 stiffness
    //   K = kappa_face * w w^T,  w_coarse = 1, w_fine = -1/m   (m fine cells)
    // → kappa_face = 4/3 (2D); diag_coarse += 4/3, diag_fine += 1/3,
    //   coarse-fine = -2/3, fine-fine = +1/3. PSD ⇒ operator stays SPD.
    static void stampCoarseFine(const AdaptiveGrid2D& grid, std::vector<std::map<int, double>>& rows,
                                int coarse, int ni, int nj, int s) {
        int cs[2][2];
        AdaptiveGrid2D::childrenOnFace(ni, nj, s, cs);
        int l = grid.dof_level[coarse];
        int fl = l + 1;
        int a[2];
        for (int t = 0; t < 2; ++t) a[t] = grid.dofAt(fl, cs[t][0], cs[t][1]);
        const int m = 2;
        const double kf = 4.0 / 3.0;
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

    // Build a Poisson RHS  b_C = V_C f_C  (+ Dirichlet shifts kappa*gd) for a
    // forcing f(x,y) and Dirichlet boundary value gd(x,y).
    std::vector<double> rhs(const std::function<double(double, double)>& f,
                            const std::function<double(double, double)>& gd) const {
        int n = g->ndof;
        std::vector<double> b(n, 0.0);
        for (int d = 0; d < n; ++d) {
            int l = g->dof_level[d], i = g->dof_i[d], j = g->dof_j[d];
            double x = g->cx(l, i), y = g->cy(l, j);
            b[d] = vol[d] * f(x, y);
            for (auto& fc : dir[d]) {
                // ghost coordinate is the face-centre (half a cell out)
                double gx = x, gy = y;
                double hh = 0.5 * g->lev[l].h;
                switch (fc.side) {
                    case 0: gx -= hh; break;
                    case 1: gx += hh; break;
                    case 2: gy -= hh; break;
                    default: gy += hh; break;
                }
                b[d] += fc.kappa * gd(gx, gy);
            }
        }
        return b;
    }

    // Sample a field at DOF cell centres.
    std::vector<double> sample(const std::function<double(double, double)>& f) const {
        int n = g->ndof;
        std::vector<double> v(n);
        for (int d = 0; d < n; ++d)
            v[d] = f(g->cx(g->dof_level[d], g->dof_i[d]), g->cy(g->dof_level[d], g->dof_j[d]));
        return v;
    }

    // Volume-weighted RMS of (a-b) over all leaves (eq. 18).
    double rmsV(const std::vector<double>& a, const std::vector<double>& b) const {
        double num = 0.0, den = 0.0;
        for (int d = 0; d < g->ndof; ++d) {
            double e = a[d] - b[d];
            num += vol[d] * e * e;
            den += vol[d];
        }
        return std::sqrt(num / den);
    }

    // Numerical Laplacian -∇²p at each leaf = (A p)/V.
    std::vector<double> numericalLaplacian(const std::vector<double>& p) const {
        std::vector<double> Ap;
        A.matvec(p, Ap);
        for (int d = 0; d < g->ndof; ++d) Ap[d] /= vol[d];
        return Ap;
    }
};

// ── Independent textbook 5-point Laplacian on a UNIFORM grid (no adaptivity) ──
// Built without any reference to AdaptiveGrid2D::couplings — used as an
// independent ground truth for the operator on uniform grids.
inline CSR uniformLaplacian5pt(int n, std::array<BC, 4> bc) {
    int N = n * n;
    auto id = [&](int i, int j) { return i + j * n; };
    std::vector<std::vector<std::pair<int, double>>> off(N);
    std::vector<double> diagv(N, 0.0);
    const int di[4] = {-1, 1, 0, 0};
    const int dj[4] = {0, 0, -1, 1};
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int r = id(i, j);
            for (int s = 0; s < 4; ++s) {
                int ni = i + di[s], nj = j + dj[s];
                if (ni < 0 || ni >= n || nj < 0 || nj >= n) {
                    if (bc[s] == BC::Dirichlet) diagv[r] += 2.0;  // kappa = h/(h/2)=2
                    continue;
                }
                diagv[r] += 1.0;
                off[r].push_back({id(ni, nj), -1.0});
            }
        }
    return buildCSR(N, off, diagv);
}

}  // namespace semistruct

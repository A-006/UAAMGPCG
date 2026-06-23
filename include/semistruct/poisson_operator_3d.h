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
    // Active (fluid) DOF list — index == CSR row. Non-cut: every leaf; cut-cell:
    // excludes solid leaves. The multigrid and rhs/sample helpers read these.
    std::vector<int> node_level, node_i, node_j, node_k;

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
        node_level = grid.dof_level; node_i = grid.dof_i; node_j = grid.dof_j; node_k = grid.dof_k;
    }

    // ── Cut-cell build (mirror of PoissonOperator2D::buildCut) ───────────────
    // phiSolid(x,y,z) < 0 inside the solid obstacle (Neumann). Fluid leaves
    // (centre outside solid) are the DOFs; each fluid face is weighted by its
    // fluid AREA fraction. Domain sides marked Dirichlet act as the air (p=0)
    // interface. Same-level faces only (uniform grids). SPD; algebraically
    // consistent (Galerkin) coarsening stays robust on cut cells (Sec.4.4/Fig.14).
    void buildCut(const AdaptiveGrid3D& grid,
                  const std::function<double(double, double, double)>& phiSolid) {
        g = &grid;
        std::vector<int> gridToActive(grid.ndof, -1);
        node_level.clear(); node_i.clear(); node_j.clear(); node_k.clear();
        for (int d = 0; d < grid.ndof; ++d) {
            int l = grid.dof_level[d], i = grid.dof_i[d], j = grid.dof_j[d], k = grid.dof_k[d];
            if (phiSolid(grid.cx(l, i), grid.cy(l, j), grid.cz(l, k)) >= 0.0) {  // fluid
                gridToActive[d] = (int)node_level.size();
                node_level.push_back(l); node_i.push_back(i); node_j.push_back(j); node_k.push_back(k);
            }
        }
        int n = (int)node_level.size();
        vol.assign(n, 0.0);
        dir.assign(n, {});
        std::vector<std::map<int, double>> rows(n);
        for (int a = 0; a < n; ++a) {
            int l = node_level[a], i = node_i[a], j = node_j[a], k = node_k[a];
            double h = grid.lev[l].h;
            vol[a] = h * h * h;
            for (int s = 0; s < 6; ++s) {
                int ni = i + AdaptiveGrid3D::FDI[s], nj = j + AdaptiveGrid3D::FDJ[s],
                    nk = k + AdaptiveGrid3D::FDK[s];
                double frac = faceFluidFraction3D(grid, phiSolid, l, i, j, k, s);
                if (!grid.lev[l].in(ni, nj, nk)) {                 // domain boundary
                    if (grid.bc[s] == BC::Dirichlet && frac > 0) {
                        double kappa = frac * (h / (0.5 * h)) * h; // frac * 2h
                        rows[a][a] += kappa;
                        dir[a].push_back({kappa, s});
                    }
                    continue;                                      // Neumann wall
                }
                if (grid.lev[l].at(ni, nj, nk) != Cell::LEAF) continue;  // (cut: uniform only)
                int b = gridToActive[grid.dofAt(l, ni, nj, nk)];
                if (b < 0) continue;                               // neighbour solid → Neumann
                if (s == 0 || s == 2 || s == 4) {                  // stamp once
                    double kappa = frac * h;                       // frac * κ_same(=h)
                    if (kappa > 0) {
                        rows[a][a] += kappa; rows[b][b] += kappa;
                        rows[a][b] -= kappa; rows[b][a] -= kappa;
                    }
                }
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

    // Fluid area fraction of the face on side s of cell (l,i,j,k): the face is a
    // square; sample phiSolid at its 4 corners and estimate the area fraction
    // with phi>=0. Exact (0 or 1) for grid-aligned cuts; for a partially cut face
    // we average the four corner indicators weighted by the two opposite-edge
    // linear fractions (marching-squares-consistent, monotone in [0,1]).
    static double faceFluidFraction3D(const AdaptiveGrid3D& grid,
                                      const std::function<double(double, double, double)>& phiSolid,
                                      int l, int i, int j, int k, int s) {
        double h = grid.lev[l].h;
        double x0 = i * h, y0 = j * h, z0 = k * h, x1 = (i + 1) * h, y1 = (j + 1) * h, z1 = (k + 1) * h;
        // The 4 corners of the face (fixed axis depends on s).
        double cx_[4], cy_[4], cz_[4];
        switch (s) {
            case 0: case 1: {  // ±x face: fixed x, spans (y,z)
                double xf = (s == 0) ? x0 : x1;
                cx_[0]=cx_[1]=cx_[2]=cx_[3]=xf;
                cy_[0]=y0; cz_[0]=z0; cy_[1]=y1; cz_[1]=z0; cy_[2]=y0; cz_[2]=z1; cy_[3]=y1; cz_[3]=z1; break;
            }
            case 2: case 3: {  // ±y face: fixed y, spans (x,z)
                double yf = (s == 2) ? y0 : y1;
                cy_[0]=cy_[1]=cy_[2]=cy_[3]=yf;
                cx_[0]=x0; cz_[0]=z0; cx_[1]=x1; cz_[1]=z0; cx_[2]=x0; cz_[2]=z1; cx_[3]=x1; cz_[3]=z1; break;
            }
            default: {         // ±z face: fixed z, spans (x,y)
                double zf = (s == 4) ? z0 : z1;
                cz_[0]=cz_[1]=cz_[2]=cz_[3]=zf;
                cx_[0]=x0; cy_[0]=y0; cx_[1]=x1; cy_[1]=y0; cx_[2]=x0; cy_[2]=y1; cx_[3]=x1; cy_[3]=y1; break;
            }
        }
        double p[4];
        int nf = 0;
        for (int c = 0; c < 4; ++c) { p[c] = phiSolid(cx_[c], cy_[c], cz_[c]); if (p[c] >= 0) nf++; }
        if (nf == 4) return 1.0;
        if (nf == 0) return 0.0;
        // partial: bilinear positive-area estimate via tensor of edge fractions.
        // corners are ordered (0,0),(1,0),(0,1),(1,1) in the two spanning axes.
        auto edgeFrac = [](double a, double b) -> double {
            bool fa = a >= 0, fb = b >= 0;
            if (fa && fb) return 1.0; if (!fa && !fb) return 0.0;
            return fa ? a / (a - b) : b / (b - a);
        };
        // average the two opposite edges along each spanning axis, then the
        // product gives a separable area fraction (exact for axis-aligned cuts).
        double fu = 0.5 * (edgeFrac(p[0], p[1]) + edgeFrac(p[2], p[3]));  // along axis-u
        double fv = 0.5 * (edgeFrac(p[0], p[2]) + edgeFrac(p[1], p[3]));  // along axis-v
        return fu * fv;
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
        int n = A.n;
        std::vector<double> b(n, 0.0);
        for (int d = 0; d < n; ++d) {
            int l = node_level[d], i = node_i[d], j = node_j[d], k = node_k[d];
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
        int n = A.n;
        std::vector<double> v(n);
        for (int d = 0; d < n; ++d)
            v[d] = f(g->cx(node_level[d], node_i[d]), g->cy(node_level[d], node_j[d]),
                     g->cz(node_level[d], node_k[d]));
        return v;
    }

    double rmsV(const std::vector<double>& a, const std::vector<double>& b) const {
        double num = 0.0, den = 0.0;
        for (int d = 0; d < A.n; ++d) {
            double e = a[d] - b[d];
            num += vol[d] * e * e; den += vol[d];
        }
        return std::sqrt(num / den);
    }

    std::vector<double> numericalLaplacian(const std::vector<double>& p) const {
        std::vector<double> Ap;
        A.matvec(p, Ap);
        for (int d = 0; d < A.n; ++d) Ap[d] /= vol[d];
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

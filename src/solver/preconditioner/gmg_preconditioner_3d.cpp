/**
 * @file gmg_preconditioner_3d.cpp
 * @brief 3D GMG V-cycle -- symmetric V-cycle, column-major indexing.
 * @author liutao
 * @date 2026-05-24
 */
#include "solver/preconditioner/gmg_preconditioner_3d.h"
#include <algorithm>
#include <cmath>

/// 3D column-major indexing: i + j*(nx+2) + k*(nx+2)*(ny+2)
static inline int cidx(int i, int j, int k, int nx, int ny) {
    int si = nx + 2;
    return i + j * si + k * si * (ny + 2);
}

void GMGPreconditioner3D::buildHierarchy(const Grid3D& fine) {
    levels_.clear();
    int nx = fine.nx, ny = fine.ny, nz = fine.nz;
    double dx = fine.dx, dy = fine.dy, dz = fine.dz;
    while (nx >= 2 && ny >= 2 && nz >= 2) {
        Level L;
        L.nx = nx; L.ny = ny; L.nz = nz;
        L.dx = dx; L.dy = dy; L.dz = dz;
        int N = (nx + 2) * (ny + 2) * (nz + 2);
        L.p.resize(N, 0.0);
        L.b.resize(N, 0.0);
        L.solid.resize(N, false);
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4) break;
        nx /= 2; ny /= 2; nz /= 2;
        dx *= 2.0; dy *= 2.0; dz *= 2.0;
    }
}

void GMGPreconditioner3D::restrictSolid(const Level& fine, Level& coarse) {
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++)
            for (int kc = 1; kc <= coarse.nz; kc++) {
                int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1;
                int sc = 0;
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        for (int dk = 0; dk < 2; dk++)
                            if (fine.solid[cidx(i_f + di, j_f + dj, k_f + dk, fine.nx, fine.ny)])
                                sc++;
                coarse.solid[cidx(ic, jc, kc, coarse.nx, coarse.ny)] = (sc >= 4);
            }
}

void GMGPreconditioner3D::smooth(Level& L, int sweeps) {
    double idx2 = 1.0 / (L.dx * L.dx);
    double idy2 = 1.0 / (L.dy * L.dy);
    double idz2 = 1.0 / (L.dz * L.dz);
    double diag = 2.0 * (idx2 + idy2 + idz2);

    for (int sw = 0; sw < sweeps; sw++) {
        // Forward pass: update cells where (i+j+k) is even.
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1; j <= L.ny; j++)
                for (int k = 1; k <= L.nz; k++) {
                    if (((i + j + k) & 1) != 0) continue;
                    int id = cidx(i, j, k, L.nx, L.ny);
                    if (L.solid[id]) continue;

                    double pL = (i > 1 && !L.solid[cidx(i - 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i - 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pR = (i < L.nx && !L.solid[cidx(i + 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i + 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pB = (j > 1 && !L.solid[cidx(i, j - 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j - 1, k, L.nx, L.ny)] : L.p[id];
                    double pT = (j < L.ny && !L.solid[cidx(i, j + 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j + 1, k, L.nx, L.ny)] : L.p[id];
                    double pF = (k > 1 && !L.solid[cidx(i, j, k - 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k - 1, L.nx, L.ny)] : L.p[id];
                    double pK = (k < L.nz && !L.solid[cidx(i, j, k + 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k + 1, L.nx, L.ny)] : L.p[id];

                    double lap = (pL + pR) * idx2 + (pB + pT) * idy2 + (pF + pK) * idz2;
                    double eff_d = diag;
                    if (i == 1   || L.solid[cidx(i - 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (i == L.nx || L.solid[cidx(i + 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[cidx(i, j - 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (j == L.ny || L.solid[cidx(i, j + 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[cidx(i, j, k - 1, L.nx, L.ny)]) eff_d -= idz2;
                    if (k == L.nz || L.solid[cidx(i, j, k + 1, L.nx, L.ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag * L.p[id] + lap);
                }

        // Backward pass: update cells where (i+j+k) is odd.
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1; j <= L.ny; j++)
                for (int k = 1; k <= L.nz; k++) {
                    if (((i + j + k) & 1) != 1) continue;
                    int id = cidx(i, j, k, L.nx, L.ny);
                    if (L.solid[id]) continue;

                    double pL = (i > 1 && !L.solid[cidx(i - 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i - 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pR = (i < L.nx && !L.solid[cidx(i + 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i + 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pB = (j > 1 && !L.solid[cidx(i, j - 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j - 1, k, L.nx, L.ny)] : L.p[id];
                    double pT = (j < L.ny && !L.solid[cidx(i, j + 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j + 1, k, L.nx, L.ny)] : L.p[id];
                    double pF = (k > 1 && !L.solid[cidx(i, j, k - 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k - 1, L.nx, L.ny)] : L.p[id];
                    double pK = (k < L.nz && !L.solid[cidx(i, j, k + 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k + 1, L.nx, L.ny)] : L.p[id];

                    double lap = (pL + pR) * idx2 + (pB + pT) * idy2 + (pF + pK) * idz2;
                    double eff_d = diag;
                    if (i == 1   || L.solid[cidx(i - 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (i == L.nx || L.solid[cidx(i + 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[cidx(i, j - 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (j == L.ny || L.solid[cidx(i, j + 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[cidx(i, j, k - 1, L.nx, L.ny)]) eff_d -= idz2;
                    if (k == L.nz || L.solid[cidx(i, j, k + 1, L.nx, L.ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag * L.p[id] + lap);
                }
    }
}

void GMGPreconditioner3D::smoothReverse(Level& L, int sweeps) {
    double idx2 = 1.0 / (L.dx * L.dx);
    double idy2 = 1.0 / (L.dy * L.dy);
    double idz2 = 1.0 / (L.dz * L.dz);
    double diag = 2.0 * (idx2 + idy2 + idz2);

    for (int sw = 0; sw < sweeps; sw++) {
        // Forward (reversed): update cells where (i+j+k) is odd.
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1; j <= L.ny; j++)
                for (int k = 1; k <= L.nz; k++) {
                    if (((i + j + k) & 1) != 1) continue;
                    int id = cidx(i, j, k, L.nx, L.ny);
                    if (L.solid[id]) continue;

                    double pL = (i > 1 && !L.solid[cidx(i - 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i - 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pR = (i < L.nx && !L.solid[cidx(i + 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i + 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pB = (j > 1 && !L.solid[cidx(i, j - 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j - 1, k, L.nx, L.ny)] : L.p[id];
                    double pT = (j < L.ny && !L.solid[cidx(i, j + 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j + 1, k, L.nx, L.ny)] : L.p[id];
                    double pF = (k > 1 && !L.solid[cidx(i, j, k - 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k - 1, L.nx, L.ny)] : L.p[id];
                    double pK = (k < L.nz && !L.solid[cidx(i, j, k + 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k + 1, L.nx, L.ny)] : L.p[id];

                    double lap = (pL + pR) * idx2 + (pB + pT) * idy2 + (pF + pK) * idz2;
                    double eff_d = diag;
                    if (i == 1   || L.solid[cidx(i - 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (i == L.nx || L.solid[cidx(i + 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[cidx(i, j - 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (j == L.ny || L.solid[cidx(i, j + 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[cidx(i, j, k - 1, L.nx, L.ny)]) eff_d -= idz2;
                    if (k == L.nz || L.solid[cidx(i, j, k + 1, L.nx, L.ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag * L.p[id] + lap);
                }

        // Backward (reversed): update cells where (i+j+k) is even.
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1; j <= L.ny; j++)
                for (int k = 1; k <= L.nz; k++) {
                    if (((i + j + k) & 1) != 0) continue;
                    int id = cidx(i, j, k, L.nx, L.ny);
                    if (L.solid[id]) continue;

                    double pL = (i > 1 && !L.solid[cidx(i - 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i - 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pR = (i < L.nx && !L.solid[cidx(i + 1, j, k, L.nx, L.ny)])
                                    ? L.p[cidx(i + 1, j, k, L.nx, L.ny)] : L.p[id];
                    double pB = (j > 1 && !L.solid[cidx(i, j - 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j - 1, k, L.nx, L.ny)] : L.p[id];
                    double pT = (j < L.ny && !L.solid[cidx(i, j + 1, k, L.nx, L.ny)])
                                    ? L.p[cidx(i, j + 1, k, L.nx, L.ny)] : L.p[id];
                    double pF = (k > 1 && !L.solid[cidx(i, j, k - 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k - 1, L.nx, L.ny)] : L.p[id];
                    double pK = (k < L.nz && !L.solid[cidx(i, j, k + 1, L.nx, L.ny)])
                                    ? L.p[cidx(i, j, k + 1, L.nx, L.ny)] : L.p[id];

                    double lap = (pL + pR) * idx2 + (pB + pT) * idy2 + (pF + pK) * idz2;
                    double eff_d = diag;
                    if (i == 1   || L.solid[cidx(i - 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (i == L.nx || L.solid[cidx(i + 1, j, k, L.nx, L.ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[cidx(i, j - 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (j == L.ny || L.solid[cidx(i, j + 1, k, L.nx, L.ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[cidx(i, j, k - 1, L.nx, L.ny)]) eff_d -= idz2;
                    if (k == L.nz || L.solid[cidx(i, j, k + 1, L.nx, L.ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag * L.p[id] + lap);
                }
    }
}

void GMGPreconditioner3D::restrictResidual(const Level& fine, Level& coarse) {
    double idx2 = 1.0 / (fine.dx * fine.dx);
    double idy2 = 1.0 / (fine.dy * fine.dy);
    double idz2 = 1.0 / (fine.dz * fine.dz);
    double diag = 2.0 * (idx2 + idy2 + idz2);

    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++)
            for (int kc = 1; kc <= coarse.nz; kc++) {
                int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1;
                double sum = 0.0;
                int cnt = 0;

                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        for (int dk = 0; dk < 2; dk++) {
                            int fi = i_f + di, fj = j_f + dj, fk = k_f + dk;
                            int fidx = cidx(fi, fj, fk, fine.nx, fine.ny);
                            if (fine.solid[fidx]) continue;

                            double pL = (fi > 1 && !fine.solid[cidx(fi - 1, fj, fk, fine.nx, fine.ny)])
                                            ? fine.p[cidx(fi - 1, fj, fk, fine.nx, fine.ny)] : fine.p[fidx];
                            double pR = (fi < fine.nx && !fine.solid[cidx(fi + 1, fj, fk, fine.nx, fine.ny)])
                                            ? fine.p[cidx(fi + 1, fj, fk, fine.nx, fine.ny)] : fine.p[fidx];
                            double pB = (fj > 1 && !fine.solid[cidx(fi, fj - 1, fk, fine.nx, fine.ny)])
                                            ? fine.p[cidx(fi, fj - 1, fk, fine.nx, fine.ny)] : fine.p[fidx];
                            double pT = (fj < fine.ny && !fine.solid[cidx(fi, fj + 1, fk, fine.nx, fine.ny)])
                                            ? fine.p[cidx(fi, fj + 1, fk, fine.nx, fine.ny)] : fine.p[fidx];
                            double pF = (fk > 1 && !fine.solid[cidx(fi, fj, fk - 1, fine.nx, fine.ny)])
                                            ? fine.p[cidx(fi, fj, fk - 1, fine.nx, fine.ny)] : fine.p[fidx];
                            double pK = (fk < fine.nz && !fine.solid[cidx(fi, fj, fk + 1, fine.nx, fine.ny)])
                                            ? fine.p[cidx(fi, fj, fk + 1, fine.nx, fine.ny)] : fine.p[fidx];

                            double lap = (pL + pR) * idx2 + (pB + pT) * idy2 + (pF + pK) * idz2;
                            sum += fine.b[fidx] - diag * fine.p[fidx] + lap;
                            cnt++;
                        }

                int cid = cidx(ic, jc, kc, coarse.nx, coarse.ny);
                if (!coarse.solid[cid] && cnt > 0)
                    coarse.b[cid] = sum / cnt;
            }
}

void GMGPreconditioner3D::prolongateAdd(const Level& coarse, Level& fine) {
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++)
            for (int kc = 1; kc <= coarse.nz; kc++) {
                int cid = cidx(ic, jc, kc, coarse.nx, coarse.ny);
                if (coarse.solid[cid]) continue;
                double val = coarse.p[cid];
                int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1;
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        for (int dk = 0; dk < 2; dk++)
                            if (!fine.solid[cidx(i_f + di, j_f + dj, k_f + dk, fine.nx, fine.ny)])
                                fine.p[cidx(i_f + di, j_f + dj, k_f + dk, fine.nx, fine.ny)] += val;
            }
}

void GMGPreconditioner3D::vCycle(int level, int nlevels) {
    Level& L = levels_[level];
    if (level == nlevels - 1) {
        // Coarsest level: 20 RBGS sweeps (10 iterations of forward+reverse).
        for (int s = 0; s < 10; s++) { smooth(L, 1); smoothReverse(L, 1); }
        return;
    }
    smooth(L, 2);
    Level& coarse = levels_[level + 1];
    restrictResidual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);
    vCycle(level + 1, nlevels);
    prolongateAdd(coarse, L);
    smoothReverse(L, 2);
}

void GMGPreconditioner3D::apply(const Grid3D& g, const std::vector<double>& r,
                                 std::vector<double>& z) {
    if (cached_nx_ != g.nx || cached_ny_ != g.ny || cached_nz_ != g.nz) {
        buildHierarchy(g);
        cached_nx_ = g.nx; cached_ny_ = g.ny; cached_nz_ = g.nz;
    }

    int nx = g.nx, ny = g.ny, nz = g.nz, nl = (int)levels_.size();

    // Copy solid mask from grid to finest level.
    for (int i = 0; i <= nx + 1; i++)
        for (int j = 0; j <= ny + 1; j++)
            for (int k = 0; k <= nz + 1; k++)
                levels_[0].solid[cidx(i, j, k, nx, ny)] = g.is_solid(i, j, k);

    // Restrict solid mask down the hierarchy.
    for (int l = 1; l < nl; l++)
        restrictSolid(levels_[l - 1], levels_[l]);

    // Copy residual RHS to finest level.
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                levels_[0].b[cidx(i, j, k, nx, ny)] = r[g.ip(i, j, k)];

    // Zero initial guess on all levels.
    for (int l = 0; l < nl; l++)
        std::fill(levels_[l].p.begin(), levels_[l].p.end(), 0.0);

    // Run V-cycle.
    vCycle(0, nl);

    // Copy solution back.
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                z[g.ip(i, j, k)] = levels_[0].p[cidx(i, j, k, nx, ny)];
}

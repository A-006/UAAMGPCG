/**
 * @file amg_preconditioner_3d.cpp
 * @brief 3D Algebraic Multigrid V-cycle -- symmetric, column-major indexing.
 * @author liutao
 * @date 2026-05-24
 */
#include "solver/preconditioner/amg_preconditioner_3d.h"
#include <algorithm>
#include <cmath>

/// 3D column-major indexing: i + j*(nx+2) + k*(nx+2)*(ny+2)
static inline int aidx(int i, int j, int k, int nx, int ny) {
    int si = nx + 2;
    return i + j * si + k * si * (ny + 2);
}

void AMGPreconditioner3D::smooth(AggLevel& L, int sweeps) {
    int nx = L.nx, ny = L.ny, nz = L.nz;
    double idx2 = L.idx2, idy2 = L.idy2, idz2 = L.idz2, diag_full = L.diag;

    for (int sw = 0; sw < sweeps; sw++) {
        // Forward pass: update cells where (i+j+k) is even.
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++) {
                    if (((i + j + k) & 1) != 0) continue;
                    int id = aidx(i, j, k, nx, ny);
                    if (L.solid[id]) continue;

                    double xL = (i > 1 && !L.solid[aidx(i - 1, j, k, nx, ny)])
                                    ? L.p[aidx(i - 1, j, k, nx, ny)] : L.p[id];
                    double xR = (i < nx && !L.solid[aidx(i + 1, j, k, nx, ny)])
                                    ? L.p[aidx(i + 1, j, k, nx, ny)] : L.p[id];
                    double xB = (j > 1 && !L.solid[aidx(i, j - 1, k, nx, ny)])
                                    ? L.p[aidx(i, j - 1, k, nx, ny)] : L.p[id];
                    double xT = (j < ny && !L.solid[aidx(i, j + 1, k, nx, ny)])
                                    ? L.p[aidx(i, j + 1, k, nx, ny)] : L.p[id];
                    double xF = (k > 1 && !L.solid[aidx(i, j, k - 1, nx, ny)])
                                    ? L.p[aidx(i, j, k - 1, nx, ny)] : L.p[id];
                    double xK = (k < nz && !L.solid[aidx(i, j, k + 1, nx, ny)])
                                    ? L.p[aidx(i, j, k + 1, nx, ny)] : L.p[id];

                    double lap = (xL + xR) * idx2 + (xB + xT) * idy2 + (xF + xK) * idz2;
                    double eff_d = diag_full;
                    if (i == 1   || L.solid[aidx(i - 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (i == nx  || L.solid[aidx(i + 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[aidx(i, j - 1, k, nx, ny)]) eff_d -= idy2;
                    if (j == ny  || L.solid[aidx(i, j + 1, k, nx, ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[aidx(i, j, k - 1, nx, ny)]) eff_d -= idz2;
                    if (k == nz  || L.solid[aidx(i, j, k + 1, nx, ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag_full * L.p[id] + lap);
                }

        // Backward pass: update cells where (i+j+k) is odd.
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++) {
                    if (((i + j + k) & 1) != 1) continue;
                    int id = aidx(i, j, k, nx, ny);
                    if (L.solid[id]) continue;

                    double xL = (i > 1 && !L.solid[aidx(i - 1, j, k, nx, ny)])
                                    ? L.p[aidx(i - 1, j, k, nx, ny)] : L.p[id];
                    double xR = (i < nx && !L.solid[aidx(i + 1, j, k, nx, ny)])
                                    ? L.p[aidx(i + 1, j, k, nx, ny)] : L.p[id];
                    double xB = (j > 1 && !L.solid[aidx(i, j - 1, k, nx, ny)])
                                    ? L.p[aidx(i, j - 1, k, nx, ny)] : L.p[id];
                    double xT = (j < ny && !L.solid[aidx(i, j + 1, k, nx, ny)])
                                    ? L.p[aidx(i, j + 1, k, nx, ny)] : L.p[id];
                    double xF = (k > 1 && !L.solid[aidx(i, j, k - 1, nx, ny)])
                                    ? L.p[aidx(i, j, k - 1, nx, ny)] : L.p[id];
                    double xK = (k < nz && !L.solid[aidx(i, j, k + 1, nx, ny)])
                                    ? L.p[aidx(i, j, k + 1, nx, ny)] : L.p[id];

                    double lap = (xL + xR) * idx2 + (xB + xT) * idy2 + (xF + xK) * idz2;
                    double eff_d = diag_full;
                    if (i == 1   || L.solid[aidx(i - 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (i == nx  || L.solid[aidx(i + 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[aidx(i, j - 1, k, nx, ny)]) eff_d -= idy2;
                    if (j == ny  || L.solid[aidx(i, j + 1, k, nx, ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[aidx(i, j, k - 1, nx, ny)]) eff_d -= idz2;
                    if (k == nz  || L.solid[aidx(i, j, k + 1, nx, ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag_full * L.p[id] + lap);
                }
    }
}

void AMGPreconditioner3D::smoothReverse(AggLevel& L, int sweeps) {
    int nx = L.nx, ny = L.ny, nz = L.nz;
    double idx2 = L.idx2, idy2 = L.idy2, idz2 = L.idz2, diag_full = L.diag;

    for (int sw = 0; sw < sweeps; sw++) {
        // Forward (reversed): update cells where (i+j+k) is odd.
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++) {
                    if (((i + j + k) & 1) != 1) continue;
                    int id = aidx(i, j, k, nx, ny);
                    if (L.solid[id]) continue;

                    double xL = (i > 1 && !L.solid[aidx(i - 1, j, k, nx, ny)])
                                    ? L.p[aidx(i - 1, j, k, nx, ny)] : L.p[id];
                    double xR = (i < nx && !L.solid[aidx(i + 1, j, k, nx, ny)])
                                    ? L.p[aidx(i + 1, j, k, nx, ny)] : L.p[id];
                    double xB = (j > 1 && !L.solid[aidx(i, j - 1, k, nx, ny)])
                                    ? L.p[aidx(i, j - 1, k, nx, ny)] : L.p[id];
                    double xT = (j < ny && !L.solid[aidx(i, j + 1, k, nx, ny)])
                                    ? L.p[aidx(i, j + 1, k, nx, ny)] : L.p[id];
                    double xF = (k > 1 && !L.solid[aidx(i, j, k - 1, nx, ny)])
                                    ? L.p[aidx(i, j, k - 1, nx, ny)] : L.p[id];
                    double xK = (k < nz && !L.solid[aidx(i, j, k + 1, nx, ny)])
                                    ? L.p[aidx(i, j, k + 1, nx, ny)] : L.p[id];

                    double lap = (xL + xR) * idx2 + (xB + xT) * idy2 + (xF + xK) * idz2;
                    double eff_d = diag_full;
                    if (i == 1   || L.solid[aidx(i - 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (i == nx  || L.solid[aidx(i + 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[aidx(i, j - 1, k, nx, ny)]) eff_d -= idy2;
                    if (j == ny  || L.solid[aidx(i, j + 1, k, nx, ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[aidx(i, j, k - 1, nx, ny)]) eff_d -= idz2;
                    if (k == nz  || L.solid[aidx(i, j, k + 1, nx, ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag_full * L.p[id] + lap);
                }

        // Backward (reversed): update cells where (i+j+k) is even.
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++) {
                    if (((i + j + k) & 1) != 0) continue;
                    int id = aidx(i, j, k, nx, ny);
                    if (L.solid[id]) continue;

                    double xL = (i > 1 && !L.solid[aidx(i - 1, j, k, nx, ny)])
                                    ? L.p[aidx(i - 1, j, k, nx, ny)] : L.p[id];
                    double xR = (i < nx && !L.solid[aidx(i + 1, j, k, nx, ny)])
                                    ? L.p[aidx(i + 1, j, k, nx, ny)] : L.p[id];
                    double xB = (j > 1 && !L.solid[aidx(i, j - 1, k, nx, ny)])
                                    ? L.p[aidx(i, j - 1, k, nx, ny)] : L.p[id];
                    double xT = (j < ny && !L.solid[aidx(i, j + 1, k, nx, ny)])
                                    ? L.p[aidx(i, j + 1, k, nx, ny)] : L.p[id];
                    double xF = (k > 1 && !L.solid[aidx(i, j, k - 1, nx, ny)])
                                    ? L.p[aidx(i, j, k - 1, nx, ny)] : L.p[id];
                    double xK = (k < nz && !L.solid[aidx(i, j, k + 1, nx, ny)])
                                    ? L.p[aidx(i, j, k + 1, nx, ny)] : L.p[id];

                    double lap = (xL + xR) * idx2 + (xB + xT) * idy2 + (xF + xK) * idz2;
                    double eff_d = diag_full;
                    if (i == 1   || L.solid[aidx(i - 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (i == nx  || L.solid[aidx(i + 1, j, k, nx, ny)]) eff_d -= idx2;
                    if (j == 1   || L.solid[aidx(i, j - 1, k, nx, ny)]) eff_d -= idy2;
                    if (j == ny  || L.solid[aidx(i, j + 1, k, nx, ny)]) eff_d -= idy2;
                    if (k == 1   || L.solid[aidx(i, j, k - 1, nx, ny)]) eff_d -= idz2;
                    if (k == nz  || L.solid[aidx(i, j, k + 1, nx, ny)]) eff_d -= idz2;

                    L.p[id] += ((eff_d < 1e-15) ? 0.0 : 1.0 / eff_d)
                               * (L.b[id] - diag_full * L.p[id] + lap);
                }
    }
}

void AMGPreconditioner3D::restrictResidual(const AggLevel& fine, AggLevel& coarse) {
    int fnx = fine.nx, fny = fine.ny, fnz = fine.nz;
    double idx2 = fine.idx2, idy2 = fine.idy2, idz2 = fine.idz2, diag_full = fine.diag;

    std::fill(coarse.b.begin(), coarse.b.end(), 0.0);
    std::vector<int> cnt(coarse.b.size(), 0);

    for (int i = 1; i <= fnx; i++)
        for (int j = 1; j <= fny; j++)
            for (int k = 1; k <= fnz; k++) {
                int idx = aidx(i, j, k, fnx, fny);
                if (fine.solid[idx]) continue;

                double xL = (i > 1 && !fine.solid[aidx(i - 1, j, k, fnx, fny)])
                                ? fine.p[aidx(i - 1, j, k, fnx, fny)] : fine.p[idx];
                double xR = (i < fnx && !fine.solid[aidx(i + 1, j, k, fnx, fny)])
                                ? fine.p[aidx(i + 1, j, k, fnx, fny)] : fine.p[idx];
                double xB = (j > 1 && !fine.solid[aidx(i, j - 1, k, fnx, fny)])
                                ? fine.p[aidx(i, j - 1, k, fnx, fny)] : fine.p[idx];
                double xT = (j < fny && !fine.solid[aidx(i, j + 1, k, fnx, fny)])
                                ? fine.p[aidx(i, j + 1, k, fnx, fny)] : fine.p[idx];
                double xF = (k > 1 && !fine.solid[aidx(i, j, k - 1, fnx, fny)])
                                ? fine.p[aidx(i, j, k - 1, fnx, fny)] : fine.p[idx];
                double xK = (k < fnz && !fine.solid[aidx(i, j, k + 1, fnx, fny)])
                                ? fine.p[aidx(i, j, k + 1, fnx, fny)] : fine.p[idx];

                double Ax = diag_full * fine.p[idx]
                          - (xL + xR) * idx2
                          - (xB + xT) * idy2
                          - (xF + xK) * idz2;
                double r_f = fine.b[idx] - Ax;

                int c = fine.agg[idx];
                if (c >= 0 && c < (int)coarse.b.size()) {
                    coarse.b[c] += r_f;
                    cnt[c]++;
                }
            }

    for (size_t k = 0; k < coarse.b.size(); k++)
        if (cnt[k] > 0) coarse.b[k] /= cnt[k];
}

void AMGPreconditioner3D::prolongateAdd(const AggLevel& coarse, AggLevel& fine) {
    int fnx = fine.nx, fny = fine.ny, fnz = fine.nz;
    for (int i = 1; i <= fnx; i++)
        for (int j = 1; j <= fny; j++)
            for (int k = 1; k <= fnz; k++) {
                int idx = aidx(i, j, k, fnx, fny);
                if (fine.solid[idx]) continue;
                int c = fine.agg[idx];
                if (c >= 0 && c < (int)coarse.p.size())
                    fine.p[idx] += coarse.p[c];
            }
}

void AMGPreconditioner3D::buildAggregates(AggLevel& fine, const AggLevel& coarse) {
    int fnx = fine.nx, fny = fine.ny, fnz = fine.nz;
    int cnx = coarse.nx, cny = coarse.ny, cnz = coarse.nz;

    for (int ic = 1; ic <= cnx; ic++)
        for (int jc = 1; jc <= cny; jc++)
            for (int kc = 1; kc <= cnz; kc++) {
                int cid = aidx(ic, jc, kc, cnx, cny);
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        for (int dk = 0; dk < 2; dk++) {
                            int fi = 2 * ic - 1 + di;
                            int fj = 2 * jc - 1 + dj;
                            int fk = 2 * kc - 1 + dk;
                            if (fi > fnx || fj > fny || fk > fnz) continue;
                            int fidx = aidx(fi, fj, fk, fnx, fny);
                            if (!fine.solid[fidx])
                                fine.agg[fidx] = cid;
                        }
            }
}

void AMGPreconditioner3D::restrictSolid(const AggLevel& fine, AggLevel& coarse) {
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++)
            for (int kc = 1; kc <= coarse.nz; kc++) {
                int i_f = 2 * ic - 1, j_f = 2 * jc - 1, k_f = 2 * kc - 1;
                int solid_count = 0, total = 0;
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        for (int dk = 0; dk < 2; dk++) {
                            int fi = i_f + di, fj = j_f + dj, fk = k_f + dk;
                            if (fi > fine.nx || fj > fine.ny || fk > fine.nz) continue;
                            total++;
                            if (fine.solid[aidx(fi, fj, fk, fine.nx, fine.ny)])
                                solid_count++;
                        }
                coarse.solid[aidx(ic, jc, kc, coarse.nx, coarse.ny)] =
                    (total > 0 && solid_count * 2 >= total);
            }
}

void AMGPreconditioner3D::buildHierarchy(const Grid3D& g) {
    levels_.clear();
    int nx = g.nx, ny = g.ny, nz = g.nz;
    double dx = g.dx, dy = g.dy, dz = g.dz;

    while (nx >= 2 && ny >= 2 && nz >= 2) {
        AggLevel L;
        L.nx = nx; L.ny = ny; L.nz = nz;
        L.dx = dx; L.dy = dy; L.dz = dz;
        L.idx2 = 1.0 / (dx * dx);
        L.idy2 = 1.0 / (dy * dy);
        L.idz2 = 1.0 / (dz * dz);
        L.diag = 2.0 * (L.idx2 + L.idy2 + L.idz2);

        int N = (nx + 2) * (ny + 2) * (nz + 2);
        L.p.resize(N, 0.0);
        L.b.resize(N, 0.0);
        L.solid.resize(N, false);
        L.agg.resize(N, -1);

        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4) break;
        nx /= 2; ny /= 2; nz /= 2;
        dx *= 2.0; dy *= 2.0; dz *= 2.0;
    }
}

void AMGPreconditioner3D::vCycle(int level, int nlevels) {
    AggLevel& L = levels_[level];
    if (level == nlevels - 1) {
        // Coarsest level: 20 RBGS sweeps (10 iterations of forward+reverse).
        for (int s = 0; s < 10; s++) { smooth(L, 1); smoothReverse(L, 1); }
        return;
    }
    smooth(L, 2);
    AggLevel& coarse = levels_[level + 1];
    restrictResidual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);
    vCycle(level + 1, nlevels);
    prolongateAdd(coarse, L);
    smoothReverse(L, 2);
}

void AMGPreconditioner3D::apply(const Grid3D& g, const std::vector<double>& r,
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
                levels_[0].solid[aidx(i, j, k, nx, ny)] = g.is_solid(i, j, k);

    // Restrict solid mask down the hierarchy.
    for (int l = 1; l < nl; l++)
        restrictSolid(levels_[l - 1], levels_[l]);

    // Build aggregates for all levels except the coarsest.
    for (int l = 0; l < nl - 1; l++)
        buildAggregates(levels_[l], levels_[l + 1]);

    // Copy residual RHS to finest level.
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                levels_[0].b[aidx(i, j, k, nx, ny)] = r[g.ip(i, j, k)];

    // Zero initial guess on all levels.
    for (int l = 0; l < nl; l++)
        std::fill(levels_[l].p.begin(), levels_[l].p.end(), 0.0);

    // Run V-cycle.
    vCycle(0, nl);

    // Copy solution back.
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                z[g.ip(i, j, k)] = levels_[0].p[aidx(i, j, k, nx, ny)];
}

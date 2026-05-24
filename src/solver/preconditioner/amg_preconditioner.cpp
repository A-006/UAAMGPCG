/**
 * @file amg_preconditioner.cpp
 * @brief Algebraic Multigrid V-cycle — symmetric, column-major indexing.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/preconditioner/amg_preconditioner.h"
#include <algorithm>
#include <cmath>

static inline int aidx(int i, int j, int stride) { return i + j * stride; }

void AMGPreconditioner::smooth(AggLevel& L, int sweeps) {
    int nx = L.nx, ny = L.ny, stride = nx + 2;
    double idx2 = L.idx2, idy2 = L.idy2, diag_full = L.diag;
    for (int s = 0; s < sweeps; s++) {
        for (int i = 1; i <= nx; i++)
            for (int j = 1 + (i%2); j <= ny; j += 2) {
                int id = aidx(i,j,stride);
                if (L.solid[id]) continue;
                double xL = (i>1 && !L.solid[aidx(i-1,j,stride)]) ? L.p[aidx(i-1,j,stride)] : L.p[id];
                double xR = (i<nx && !L.solid[aidx(i+1,j,stride)]) ? L.p[aidx(i+1,j,stride)] : L.p[id];
                double xB = (j>1 && !L.solid[aidx(i,j-1,stride)]) ? L.p[aidx(i,j-1,stride)] : L.p[id];
                double xT = (j<ny && !L.solid[aidx(i,j+1,stride)]) ? L.p[aidx(i,j+1,stride)] : L.p[id];
                double lap = (xL+xR)*idx2 + (xB+xT)*idy2;
                double eff_d = diag_full;
                if (i==1||L.solid[aidx(i-1,j,stride)]) eff_d -= idx2;
                if (i==nx||L.solid[aidx(i+1,j,stride)]) eff_d -= idx2;
                if (j==1||L.solid[aidx(i,j-1,stride)]) eff_d -= idy2;
                if (j==ny||L.solid[aidx(i,j+1,stride)]) eff_d -= idy2;
                L.p[id] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[id] - diag_full * L.p[id] + lap);
            }
        for (int i = 1; i <= nx; i++)
            for (int j = 1 + ((i+1)%2); j <= ny; j += 2) {
                int id = aidx(i,j,stride);
                if (L.solid[id]) continue;
                double xL = (i>1 && !L.solid[aidx(i-1,j,stride)]) ? L.p[aidx(i-1,j,stride)] : L.p[id];
                double xR = (i<nx && !L.solid[aidx(i+1,j,stride)]) ? L.p[aidx(i+1,j,stride)] : L.p[id];
                double xB = (j>1 && !L.solid[aidx(i,j-1,stride)]) ? L.p[aidx(i,j-1,stride)] : L.p[id];
                double xT = (j<ny && !L.solid[aidx(i,j+1,stride)]) ? L.p[aidx(i,j+1,stride)] : L.p[id];
                double lap = (xL+xR)*idx2 + (xB+xT)*idy2;
                double eff_d = diag_full;
                if (i==1||L.solid[aidx(i-1,j,stride)]) eff_d -= idx2;
                if (i==nx||L.solid[aidx(i+1,j,stride)]) eff_d -= idx2;
                if (j==1||L.solid[aidx(i,j-1,stride)]) eff_d -= idy2;
                if (j==ny||L.solid[aidx(i,j+1,stride)]) eff_d -= idy2;
                L.p[id] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[id] - diag_full * L.p[id] + lap);
            }
    }
}

void AMGPreconditioner::smoothReverse(AggLevel& L, int sweeps) {
    int nx = L.nx, ny = L.ny, stride = nx + 2;
    double idx2 = L.idx2, idy2 = L.idy2, diag_full = L.diag;
    for (int s = 0; s < sweeps; s++) {
        for (int i = 1; i <= nx; i++)
            for (int j = 1 + ((i+1)%2); j <= ny; j += 2) {
                int id = aidx(i,j,stride);
                if (L.solid[id]) continue;
                double xL = (i>1 && !L.solid[aidx(i-1,j,stride)]) ? L.p[aidx(i-1,j,stride)] : L.p[id];
                double xR = (i<nx && !L.solid[aidx(i+1,j,stride)]) ? L.p[aidx(i+1,j,stride)] : L.p[id];
                double xB = (j>1 && !L.solid[aidx(i,j-1,stride)]) ? L.p[aidx(i,j-1,stride)] : L.p[id];
                double xT = (j<ny && !L.solid[aidx(i,j+1,stride)]) ? L.p[aidx(i,j+1,stride)] : L.p[id];
                double lap = (xL+xR)*idx2 + (xB+xT)*idy2;
                double eff_d = diag_full;
                if (i==1||L.solid[aidx(i-1,j,stride)]) eff_d -= idx2;
                if (i==nx||L.solid[aidx(i+1,j,stride)]) eff_d -= idx2;
                if (j==1||L.solid[aidx(i,j-1,stride)]) eff_d -= idy2;
                if (j==ny||L.solid[aidx(i,j+1,stride)]) eff_d -= idy2;
                L.p[id] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[id] - diag_full * L.p[id] + lap);
            }
        for (int i = 1; i <= nx; i++)
            for (int j = 1 + (i%2); j <= ny; j += 2) {
                int id = aidx(i,j,stride);
                if (L.solid[id]) continue;
                double xL = (i>1 && !L.solid[aidx(i-1,j,stride)]) ? L.p[aidx(i-1,j,stride)] : L.p[id];
                double xR = (i<nx && !L.solid[aidx(i+1,j,stride)]) ? L.p[aidx(i+1,j,stride)] : L.p[id];
                double xB = (j>1 && !L.solid[aidx(i,j-1,stride)]) ? L.p[aidx(i,j-1,stride)] : L.p[id];
                double xT = (j<ny && !L.solid[aidx(i,j+1,stride)]) ? L.p[aidx(i,j+1,stride)] : L.p[id];
                double lap = (xL+xR)*idx2 + (xB+xT)*idy2;
                double eff_d = diag_full;
                if (i==1||L.solid[aidx(i-1,j,stride)]) eff_d -= idx2;
                if (i==nx||L.solid[aidx(i+1,j,stride)]) eff_d -= idx2;
                if (j==1||L.solid[aidx(i,j-1,stride)]) eff_d -= idy2;
                if (j==ny||L.solid[aidx(i,j+1,stride)]) eff_d -= idy2;
                L.p[id] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[id] - diag_full * L.p[id] + lap);
            }
    }
}

void AMGPreconditioner::restrictResidual(const AggLevel& fine, AggLevel& coarse) {
    int fnx = fine.nx, fny = fine.ny, fs = fnx + 2;
    double idx2 = fine.idx2, idy2 = fine.idy2, diag_full = fine.diag;
    std::fill(coarse.b.begin(), coarse.b.end(), 0.0);
    std::vector<int> cnt(coarse.b.size(), 0);
    for (int i = 1; i <= fnx; i++)
        for (int j = 1; j <= fny; j++) {
            int idx = aidx(i,j,fs);
            if (fine.solid[idx]) continue;
            double xL = (i>1 && !fine.solid[aidx(i-1,j,fs)]) ? fine.p[aidx(i-1,j,fs)] : fine.p[idx];
            double xR = (i<fnx && !fine.solid[aidx(i+1,j,fs)]) ? fine.p[aidx(i+1,j,fs)] : fine.p[idx];
            double xB = (j>1 && !fine.solid[aidx(i,j-1,fs)]) ? fine.p[aidx(i,j-1,fs)] : fine.p[idx];
            double xT = (j<fny && !fine.solid[aidx(i,j+1,fs)]) ? fine.p[aidx(i,j+1,fs)] : fine.p[idx];
            double Ax = diag_full * fine.p[idx] - (xL+xR)*idx2 - (xB+xT)*idy2;
            double r_f = fine.b[idx] - Ax;
            int c = fine.agg[idx];
            if (c >= 0 && c < (int)coarse.b.size()) { coarse.b[c] += r_f; cnt[c]++; }
        }
    for (size_t k = 0; k < coarse.b.size(); k++)
        if (cnt[k] > 0) coarse.b[k] /= cnt[k];
}

void AMGPreconditioner::prolongateAdd(const AggLevel& coarse, AggLevel& fine) {
    int fnx = fine.nx, fny = fine.ny, fs = fnx + 2;
    for (int i = 1; i <= fnx; i++)
        for (int j = 1; j <= fny; j++) {
            int idx = aidx(i,j,fs);
            if (fine.solid[idx]) continue;
            int c = fine.agg[idx];
            if (c >= 0 && c < (int)coarse.p.size())
                fine.p[idx] += coarse.p[c];
        }
}

void AMGPreconditioner::buildAggregates(AggLevel& fine, const AggLevel& coarse) {
    int fnx = fine.nx, fny = fine.ny, fs = fnx + 2;
    int cnx = coarse.nx, cny = coarse.ny, cs = cnx + 2;
    for (int ic = 1; ic <= cnx; ic++)
        for (int jc = 1; jc <= cny; jc++) {
            int cidx = aidx(ic, jc, cs);
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++) {
                    int fi = 2*ic - 1 + di, fj = 2*jc - 1 + dj;
                    if (fi > fnx || fj > fny) continue;
                    int fidx = aidx(fi, fj, fs);
                    if (!fine.solid[fidx]) fine.agg[fidx] = cidx;
                }
        }
}

void AMGPreconditioner::restrictSolid(const AggLevel& fine, AggLevel& coarse) {
    int fs = fine.nx + 2, cs = coarse.nx + 2;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1, solid_count = 0, total = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++) {
                    int fi = i_f + di, fj = j_f + dj;
                    if (fi > fine.nx || fj > fine.ny) continue;
                    total++;
                    if (fine.solid[aidx(fi, fj, fs)]) solid_count++;
                }
            coarse.solid[aidx(ic, jc, cs)] = (total > 0 && solid_count * 2 >= total);
        }
}

void AMGPreconditioner::buildHierarchy(const Grid& g) {
    levels_.clear();
    int nx = g.nx, ny = g.ny;
    double dx = g.dx, dy = g.dy;
    while (nx >= 2 && ny >= 2) {
        AggLevel L;
        L.nx = nx; L.ny = ny; L.dx = dx; L.dy = dy;
        L.idx2 = 1.0/(dx*dx); L.idy2 = 1.0/(dy*dy);
        L.diag = 2.0*L.idx2 + 2.0*L.idy2;
        int N = (nx+2)*(ny+2);
        L.p.resize(N, 0.0); L.b.resize(N, 0.0);
        L.solid.resize(N, false); L.agg.resize(N, -1);
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4) break;
        nx /= 2; ny /= 2; dx *= 2.0; dy *= 2.0;
    }
}

void AMGPreconditioner::vCycle(int level, int nlevels) {
    AggLevel& L = levels_[level];
    if (level == nlevels - 1) {
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

void AMGPreconditioner::apply(const Grid& g, const std::vector<double>& r,
                               std::vector<double>& z) {
    if (cached_nx_ != g.nx || cached_ny_ != g.ny) {
        buildHierarchy(g); cached_nx_ = g.nx; cached_ny_ = g.ny;
    }
    int nx = g.nx, ny = g.ny, nl = (int)levels_.size(), stride0 = nx + 2;
    for (int i = 0; i <= nx+1; i++)
        for (int j = 0; j <= ny+1; j++)
            levels_[0].solid[aidx(i,j,stride0)] = g.is_solid(i,j);
    for (int l = 1; l < nl; l++) restrictSolid(levels_[l-1], levels_[l]);
    for (int l = 0; l < nl - 1; l++) buildAggregates(levels_[l], levels_[l+1]);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            levels_[0].b[aidx(i,j,stride0)] = r[g.ip(i,j)];
    for (int l = 0; l < nl; l++) std::fill(levels_[l].p.begin(), levels_[l].p.end(), 0.0);
    vCycle(0, nl);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            z[g.ip(i,j)] = levels_[0].p[aidx(i,j,stride0)];
}

/**
 * @file uaamg_preconditioner.cpp
 * @brief UAAMG V-cycle — Algorithm 3 from Sun et al. SIGGRAPH 2025.
 *
 * 1 pre + 1 post RBGS, scale-2 prolongation (Stuben 2001),
 * 4-to-1 averaging restriction, column-major indexing.
 */
#include "solver/preconditioner/uaamg_preconditioner.h"
#include <algorithm>
#include <cmath>

static inline int cidx(int i, int j, int stride) { return i + j * stride; }

void UAAMGPreconditioner::buildHierarchy(const Grid& fine) {
    levels_.clear();
    int nx = fine.nx, ny = fine.ny;
    double dx = fine.dx, dy = fine.dy;
    while (nx >= 2 && ny >= 2) {
        Level L;
        L.nx = nx; L.ny = ny; L.dx = dx; L.dy = dy;
        L.p.resize((nx+2)*(ny+2), 0.0);
        L.b.resize((nx+2)*(ny+2), 0.0);
        L.solid.resize((nx+2)*(ny+2), false);
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4) break;
        nx /= 2; ny /= 2; dx *= 2.0; dy *= 2.0;
    }
}

void UAAMGPreconditioner::restrictSolid(const Level& fine, Level& coarse) {
    int fs = fine.nx + 2, cs = coarse.nx + 2;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1, sc = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    if (fine.solid[cidx(i_f+di, j_f+dj, fs)]) sc++;
            coarse.solid[cidx(ic, jc, cs)] = (sc >= 2);
        }
}

void UAAMGPreconditioner::smooth(Level& L, int sweeps) {
    double idx2 = 1.0/(L.dx*L.dx), idy2 = 1.0/(L.dy*L.dy);
    double diag = 2.0*(idx2+idy2);
    int stride = L.nx + 2;
    for (int s = 0; s < sweeps; s++) {
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1 + (i%2); j <= L.ny; j += 2) {
                int id = cidx(i,j,stride);
                if (L.solid[id]) continue;
                double pL = (i>1 && !L.solid[cidx(i-1,j,stride)]) ? L.p[cidx(i-1,j,stride)] : L.p[id];
                double pR = (i<L.nx && !L.solid[cidx(i+1,j,stride)]) ? L.p[cidx(i+1,j,stride)] : L.p[id];
                double pB = (j>1 && !L.solid[cidx(i,j-1,stride)]) ? L.p[cidx(i,j-1,stride)] : L.p[id];
                double pT = (j<L.ny && !L.solid[cidx(i,j+1,stride)]) ? L.p[cidx(i,j+1,stride)] : L.p[id];
                double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
                double eff_d = diag;
                if (i==1||L.solid[cidx(i-1,j,stride)]) eff_d -= idx2;
                if (i==L.nx||L.solid[cidx(i+1,j,stride)]) eff_d -= idx2;
                if (j==1||L.solid[cidx(i,j-1,stride)]) eff_d -= idy2;
                if (j==L.ny||L.solid[cidx(i,j+1,stride)]) eff_d -= idy2;
                L.p[id] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[id] - diag * L.p[id] + lap);
            }
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1 + ((i+1)%2); j <= L.ny; j += 2) {
                int id = cidx(i,j,stride);
                if (L.solid[id]) continue;
                double pL = (i>1 && !L.solid[cidx(i-1,j,stride)]) ? L.p[cidx(i-1,j,stride)] : L.p[id];
                double pR = (i<L.nx && !L.solid[cidx(i+1,j,stride)]) ? L.p[cidx(i+1,j,stride)] : L.p[id];
                double pB = (j>1 && !L.solid[cidx(i,j-1,stride)]) ? L.p[cidx(i,j-1,stride)] : L.p[id];
                double pT = (j<L.ny && !L.solid[cidx(i,j+1,stride)]) ? L.p[cidx(i,j+1,stride)] : L.p[id];
                double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
                double eff_d = diag;
                if (i==1||L.solid[cidx(i-1,j,stride)]) eff_d -= idx2;
                if (i==L.nx||L.solid[cidx(i+1,j,stride)]) eff_d -= idx2;
                if (j==1||L.solid[cidx(i,j-1,stride)]) eff_d -= idy2;
                if (j==L.ny||L.solid[cidx(i,j+1,stride)]) eff_d -= idy2;
                L.p[id] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[id] - diag * L.p[id] + lap);
            }
    }
}

void UAAMGPreconditioner::restrictResidual(const Level& fine, Level& coarse) {
    int fs = fine.nx + 2, cs = coarse.nx + 2;
    double idx2 = 1.0/(fine.dx*fine.dx), idy2 = 1.0/(fine.dy*fine.dy);
    double diag = 2.0*(idx2+idy2);
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            double sum = 0; int cnt = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++) {
                    int fi = i_f + di, fj = j_f + dj, fidx = cidx(fi, fj, fs);
                    if (fine.solid[fidx]) continue;
                    double pL = (fi>1 && !fine.solid[cidx(fi-1,fj,fs)]) ? fine.p[cidx(fi-1,fj,fs)] : fine.p[fidx];
                    double pR = (fi<fine.nx && !fine.solid[cidx(fi+1,fj,fs)]) ? fine.p[cidx(fi+1,fj,fs)] : fine.p[fidx];
                    double pB = (fj>1 && !fine.solid[cidx(fi,fj-1,fs)]) ? fine.p[cidx(fi,fj-1,fs)] : fine.p[fidx];
                    double pT = (fj<fine.ny && !fine.solid[cidx(fi,fj+1,fs)]) ? fine.p[cidx(fi,fj+1,fs)] : fine.p[fidx];
                    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
                    sum += fine.b[fidx] - diag * fine.p[fidx] + lap;
                    cnt++;
                }
            int cid = cidx(ic, jc, cs);
            if (!coarse.solid[cid] && cnt > 0) coarse.b[cid] = sum / cnt;
        }
}

void UAAMGPreconditioner::prolongateAdd(const Level& coarse, Level& fine) {
    int cs = coarse.nx + 2, fs = fine.nx + 2;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int cid = cidx(ic, jc, cs);
            if (coarse.solid[cid]) continue;
            double val = 2.0 * coarse.p[cid];  // scale factor 2 (Stuben 2001)
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    if (!fine.solid[cidx(i_f+di, j_f+dj, fs)])
                        fine.p[cidx(i_f+di, j_f+dj, fs)] += val;
        }
}

void UAAMGPreconditioner::vCycle(int level, int nlevels) {
    Level& L = levels_[level];
    if (level == nlevels - 1) { smooth(L, 20); return; }
    smooth(L, 1);
    Level& coarse = levels_[level + 1];
    restrictResidual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);
    vCycle(level + 1, nlevels);
    prolongateAdd(coarse, L);
    smooth(L, 1);
}

void UAAMGPreconditioner::apply(const Grid& g, const std::vector<double>& r,
                                 std::vector<double>& z) {
    if (cached_nx_ != g.nx || cached_ny_ != g.ny) {
        buildHierarchy(g); cached_nx_ = g.nx; cached_ny_ = g.ny;
    }
    int nx = g.nx, ny = g.ny, nl = (int)levels_.size(), stride0 = nx + 2;
    for (int i = 0; i <= nx+1; i++)
        for (int j = 0; j <= ny+1; j++)
            levels_[0].solid[cidx(i,j,stride0)] = g.is_solid(i,j);
    for (int l = 1; l < nl; l++) restrictSolid(levels_[l-1], levels_[l]);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            levels_[0].b[cidx(i,j,stride0)] = r[g.ip(i,j)];
    for (int l = 0; l < nl; l++) std::fill(levels_[l].p.begin(), levels_[l].p.end(), 0.0);
    vCycle(0, nl);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            z[g.ip(i,j)] = levels_[0].p[cidx(i,j,stride0)];
}

/**
 * @file gmg_preconditioner.cpp
 * @brief Geometric Multigrid V-cycle — geometric coarsening with RBGS smoothing.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/preconditioner/gmg_preconditioner.h"
#include <algorithm>
#include <cmath>

void GMGPreconditioner::buildHierarchy(const Grid& fine) {
    levels_.clear();
    int nx = fine.nx, ny = fine.ny;
    double dx = fine.dx, dy = fine.dy;

    while (nx >= 2 && ny >= 2) {
        Level L;
        L.nx = nx; L.ny = ny;
        L.dx = dx; L.dy = dy;
        L.p.resize((nx+2)*(ny+2), 0.0);
        L.b.resize((nx+2)*(ny+2), 0.0);
        L.solid.resize((nx+2)*(ny+2), false);
        levels_.push_back(std::move(L));

        if (nx <= 4 || ny <= 4) break;
        nx = nx / 2;
        ny = ny / 2;
        dx *= 2.0;
        dy *= 2.0;
    }
}

void GMGPreconditioner::restrictSolid(const Level& fine, Level& coarse) {
    for (int ic = 1; ic <= coarse.nx; ic++) {
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            int solid_count = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    if (fine.solid[(i_f+di)*(fine.ny+2) + (j_f+dj)])
                        solid_count++;
            coarse.solid[ic*(coarse.ny+2) + jc] = (solid_count >= 2);
        }
    }
}

void GMGPreconditioner::smooth(Level& L, int sweeps) {
    double idx2 = 1.0/(L.dx*L.dx), idy2 = 1.0/(L.dy*L.dy);
    double diag = 2.0*(idx2+idy2);
    int stride = L.ny + 2;

    for (int s = 0; s < sweeps; s++) {
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1 + (i%2); j <= L.ny; j += 2) {
                int idx = i*stride + j;
                if (L.solid[idx]) continue;
                double pL = (i>1 && !L.solid[(i-1)*stride+j]) ? L.p[(i-1)*stride+j] : L.p[idx];
                double pR = (i<L.nx && !L.solid[(i+1)*stride+j]) ? L.p[(i+1)*stride+j] : L.p[idx];
                double pB = (j>1 && !L.solid[i*stride+j-1]) ? L.p[i*stride+j-1] : L.p[idx];
                double pT = (j<L.ny && !L.solid[i*stride+j+1]) ? L.p[i*stride+j+1] : L.p[idx];
                double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
                double eff_d = diag;
                if (i==1||L.solid[(i-1)*stride+j]) eff_d -= idx2;
                if (i==L.nx||L.solid[(i+1)*stride+j]) eff_d -= idx2;
                if (j==1||L.solid[i*stride+j-1]) eff_d -= idy2;
                if (j==L.ny||L.solid[i*stride+j+1]) eff_d -= idy2;
                L.p[idx] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[idx] - lap);
            }
        for (int i = 1; i <= L.nx; i++)
            for (int j = 1 + ((i+1)%2); j <= L.ny; j += 2) {
                int idx = i*stride + j;
                if (L.solid[idx]) continue;
                double pL = (i>1 && !L.solid[(i-1)*stride+j]) ? L.p[(i-1)*stride+j] : L.p[idx];
                double pR = (i<L.nx && !L.solid[(i+1)*stride+j]) ? L.p[(i+1)*stride+j] : L.p[idx];
                double pB = (j>1 && !L.solid[i*stride+j-1]) ? L.p[i*stride+j-1] : L.p[idx];
                double pT = (j<L.ny && !L.solid[i*stride+j+1]) ? L.p[i*stride+j+1] : L.p[idx];
                double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
                double eff_d = diag;
                if (i==1||L.solid[(i-1)*stride+j]) eff_d -= idx2;
                if (i==L.nx||L.solid[(i+1)*stride+j]) eff_d -= idx2;
                if (j==1||L.solid[i*stride+j-1]) eff_d -= idy2;
                if (j==L.ny||L.solid[i*stride+j+1]) eff_d -= idy2;
                L.p[idx] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[idx] - lap);
            }
    }
}

void GMGPreconditioner::restrictResidual(const Level& fine, Level& coarse) {
    int fs = fine.ny + 2, cs = coarse.ny + 2;
    double idx2 = 1.0/(fine.dx*fine.dx), idy2 = 1.0/(fine.dy*fine.dy);
    double diag = 2.0*(idx2+idy2);

    for (int ic = 1; ic <= coarse.nx; ic++) {
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            double sum = 0; int cnt = 0;
            for (int di = 0; di < 2; di++) {
                for (int dj = 0; dj < 2; dj++) {
                    int fi = i_f + di, fj = j_f + dj;
                    int fidx = fi*fs + fj;
                    if (fine.solid[fidx]) continue;
                    double pL = (fi>1 && !fine.solid[(fi-1)*fs+fj]) ? fine.p[(fi-1)*fs+fj] : fine.p[fidx];
                    double pR = (fi<fine.nx && !fine.solid[(fi+1)*fs+fj]) ? fine.p[(fi+1)*fs+fj] : fine.p[fidx];
                    double pB = (fj>1 && !fine.solid[fi*fs+fj-1]) ? fine.p[fi*fs+fj-1] : fine.p[fidx];
                    double pT = (fj<fine.ny && !fine.solid[fi*fs+fj+1]) ? fine.p[fi*fs+fj+1] : fine.p[fidx];
                    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
                    sum += fine.b[fidx] - lap;
                    cnt++;
                }
            }
            int cidx = ic*cs + jc;
            if (!coarse.solid[cidx] && cnt > 0)
                coarse.b[cidx] = sum / cnt;
        }
    }
}

void GMGPreconditioner::prolongateAdd(const Level& coarse, Level& fine) {
    int cs = coarse.ny + 2, fs = fine.ny + 2;
    for (int ic = 1; ic <= coarse.nx; ic++) {
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int cidx = ic*cs + jc;
            if (coarse.solid[cidx]) continue;
            double val = coarse.p[cidx];
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    if (!fine.solid[(i_f+di)*fs + (j_f+dj)])
                        fine.p[(i_f+di)*fs + (j_f+dj)] += val;
        }
    }
}

void GMGPreconditioner::vCycle(int level, int nlevels) {
    Level& L = levels_[level];

    if (level == nlevels - 1) {
        smooth(L, 20);
        return;
    }

    smooth(L, 2);
    Level& coarse = levels_[level + 1];
    restrictResidual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);
    vCycle(level + 1, nlevels);
    prolongateAdd(coarse, L);
    smooth(L, 2);
}

void GMGPreconditioner::apply(const Grid& g, const std::vector<double>& r,
                               std::vector<double>& z) {
    // Rebuild hierarchy if grid size changed
    if (cached_nx_ != g.nx || cached_ny_ != g.ny) {
        buildHierarchy(g);
        cached_nx_ = g.nx;
        cached_ny_ = g.ny;
    }

    int nx = g.nx, ny = g.ny, nl = (int)levels_.size();

    // Copy solid mask to finest level
    for (int i = 0; i <= nx+1; i++)
        for (int j = 0; j <= ny+1; j++)
            levels_[0].solid[i*(ny+2)+j] = g.is_solid(i,j);

    // Propagate solid masks to coarser levels
    for (int l = 1; l < nl; l++)
        restrictSolid(levels_[l-1], levels_[l]);

    // Copy r to finest level RHS
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            levels_[0].b[g.ip(i,j)] = r[g.ip(i,j)];

    // Zero all level pressures
    for (int l = 0; l < nl; l++)
        std::fill(levels_[l].p.begin(), levels_[l].p.end(), 0.0);

    // One V-Cycle
    vCycle(0, nl);

    // Copy finest level correction to z
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            z[g.ip(i,j)] = levels_[0].p[g.ip(i,j)];
}

/**
 * @file amg_preconditioner.cpp
 * @brief Algebraic Multigrid V-cycle — aggregation-based coarsening with Galerkin coarse operators.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/preconditioner/amg_preconditioner.h"
#include <algorithm>
#include <cmath>

// ── RBGS smoothing ──
void AMGPreconditioner::smooth(AggLevel& L, int sweeps) {
    int nx = L.nx, ny = L.ny, stride = ny + 2;
    double idx2 = L.idx2, idy2 = L.idy2;

    for (int s = 0; s < sweeps; s++) {
        for (int i = 1; i <= nx; i++) {
            for (int j = 1 + (i%2); j <= ny; j += 2) {
                int idx = i*stride + j;
                if (L.solid[idx]) continue;

                double xL = (i>1 && !L.solid[(i-1)*stride+j]) ? L.p[(i-1)*stride+j] : L.p[idx];
                double xR = (i<nx && !L.solid[(i+1)*stride+j]) ? L.p[(i+1)*stride+j] : L.p[idx];
                double xB = (j>1 && !L.solid[i*stride+j-1]) ? L.p[i*stride+j-1] : L.p[idx];
                double xT = (j<ny && !L.solid[i*stride+j+1]) ? L.p[i*stride+j+1] : L.p[idx];

                double diag_full = 2.0*(idx2+idy2);
                double lap = (xL+xR)*idx2 + (xB+xT)*idy2;
                double eff_d = diag_full;

                if (i==1||L.solid[(i-1)*stride+j]) eff_d -= idx2;
                if (i==nx||L.solid[(i+1)*stride+j]) eff_d -= idx2;
                if (j==1||L.solid[i*stride+j-1]) eff_d -= idy2;
                if (j==ny||L.solid[i*stride+j+1]) eff_d -= idy2;

                L.p[idx] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[idx] - lap);
            }
        }
        for (int i = 1; i <= nx; i++) {
            for (int j = 1 + ((i+1)%2); j <= ny; j += 2) {
                int idx = i*stride + j;
                if (L.solid[idx]) continue;

                double xL = (i>1 && !L.solid[(i-1)*stride+j]) ? L.p[(i-1)*stride+j] : L.p[idx];
                double xR = (i<nx && !L.solid[(i+1)*stride+j]) ? L.p[(i+1)*stride+j] : L.p[idx];
                double xB = (j>1 && !L.solid[i*stride+j-1]) ? L.p[i*stride+j-1] : L.p[idx];
                double xT = (j<ny && !L.solid[i*stride+j+1]) ? L.p[i*stride+j+1] : L.p[idx];

                double diag_full = 2.0*(idx2+idy2);
                double lap = (xL+xR)*idx2 + (xB+xT)*idy2;
                double eff_d = diag_full;

                if (i==1||L.solid[(i-1)*stride+j]) eff_d -= idx2;
                if (i==nx||L.solid[(i+1)*stride+j]) eff_d -= idx2;
                if (j==1||L.solid[i*stride+j-1]) eff_d -= idy2;
                if (j==ny||L.solid[i*stride+j+1]) eff_d -= idy2;

                L.p[idx] += ((eff_d<1e-15)?0.0:1.0/eff_d) * (L.b[idx] - lap);
            }
        }
    }
}

// ── Restrict residual using aggregation (r_coarse = P^T * (b - A*x)) ──
void AMGPreconditioner::restrictResidual(const AggLevel& fine, AggLevel& coarse) {
    int fnx = fine.nx, fny = fine.ny, fs = fny + 2;
    double idx2 = fine.idx2, idy2 = fine.idy2;
    double diag_full = 2.0*(idx2+idy2);

    // Compute fine residual r = b - A*x and restrict in one pass
    std::fill(coarse.b.begin(), coarse.b.end(), 0.0);
    std::vector<int> cnt(coarse.b.size(), 0);

    for (int i = 1; i <= fnx; i++) {
        for (int j = 1; j <= fny; j++) {
            int idx = i*fs + j;
            if (fine.solid[idx]) continue;

            // A*x at (i,j)
            double xL = (i>1 && !fine.solid[(i-1)*fs+j]) ? fine.p[(i-1)*fs+j] : fine.p[idx];
            double xR = (i<fnx && !fine.solid[(i+1)*fs+j]) ? fine.p[(i+1)*fs+j] : fine.p[idx];
            double xB = (j>1 && !fine.solid[i*fs+j-1]) ? fine.p[i*fs+j-1] : fine.p[idx];
            double xT = (j<fny && !fine.solid[i*fs+j+1]) ? fine.p[i*fs+j+1] : fine.p[idx];
            double Ax = diag_full * fine.p[idx] - (xL+xR)*idx2 - (xB+xT)*idy2;

            double r_f = fine.b[idx] - Ax;
            int c = fine.agg[idx];
            if (c >= 0 && c < (int)coarse.b.size()) {
                coarse.b[c] += r_f;
                cnt[c]++;
            }
        }
    }

    // Average the restricted residual
    for (size_t k = 0; k < coarse.b.size(); k++)
        if (cnt[k] > 0) coarse.b[k] /= cnt[k];
}

// ── Prolongate correction using aggregation (x_fine += P * x_coarse) ──
void AMGPreconditioner::prolongateAdd(const AggLevel& coarse, AggLevel& fine) {
    int fnx = fine.nx, fny = fine.ny, fs = fny + 2;

    for (int i = 1; i <= fnx; i++) {
        for (int j = 1; j <= fny; j++) {
            int idx = i*fs + j;
            if (fine.solid[idx]) continue;
            int c = fine.agg[idx];
            if (c >= 0 && c < (int)coarse.p.size())
                fine.p[idx] += coarse.p[c];
        }
    }
}

// ── Build aggregates: map fine cells to coarse aggregates ──
void AMGPreconditioner::buildAggregates(AggLevel& fine, const AggLevel& coarse) {
    int fnx = fine.nx, fny = fine.ny, fs = fny + 2;
    int cnx = coarse.nx, cny = coarse.ny, cs = cny + 2;

    for (int ic = 1; ic <= cnx; ic++) {
        for (int jc = 1; jc <= cny; jc++) {
            int cidx = ic*cs + jc;

            bool has_fluid = false;
            for (int di = 0; di < 2; di++) {
                for (int dj = 0; dj < 2; dj++) {
                    int fi = 2*ic - 1 + di;
                    int fj = 2*jc - 1 + dj;
                    if (fi > fnx || fj > fny) continue;
                    int fidx = fi*fs + fj;
                    if (!fine.solid[fidx]) {
                        fine.agg[fidx] = cidx;
                        has_fluid = true;
                    }
                }
            }
        }
    }
}

// ── Restrict solid mask from fine to coarse (AMG version) ──
void AMGPreconditioner::restrictSolid(const AggLevel& fine, AggLevel& coarse) {
    for (int ic = 1; ic <= coarse.nx; ic++) {
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            int solid_count = 0, total = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++) {
                    int fi = i_f + di, fj = j_f + dj;
                    if (fi > fine.nx || fj > fine.ny) continue;
                    total++;
                    if (fine.solid[fi*(fine.ny+2) + fj])
                        solid_count++;
                }
            coarse.solid[ic*(coarse.ny+2) + jc] = (total > 0 && solid_count * 2 >= total);
        }
    }
}

// ── Build multi-level hierarchy ──
void AMGPreconditioner::buildHierarchy(const Grid& g) {
    levels_.clear();

    int nx = g.nx, ny = g.ny;
    double dx = g.dx, dy = g.dy;

    while (nx >= 2 && ny >= 2) {
        AggLevel L;
        L.nx = nx; L.ny = ny;
        L.dx = dx; L.dy = dy;
        L.idx2 = 1.0/(dx*dx);
        L.idy2 = 1.0/(dy*dy);
        L.diag = 2.0*L.idx2 + 2.0*L.idy2;

        int N = (nx+2)*(ny+2);
        L.p.resize(N, 0.0);
        L.b.resize(N, 0.0);
        L.solid.resize(N, false);
        L.agg.resize(N, -1);

        levels_.push_back(std::move(L));

        if (nx <= 4 || ny <= 4) break;
        nx = nx / 2;
        ny = ny / 2;
        dx *= 2.0;
        dy *= 2.0;
    }
}

// ── Recursive V-Cycle ──
void AMGPreconditioner::vCycle(int level, int nlevels) {
    AggLevel& L = levels_[level];

    if (level == nlevels - 1) {
        smooth(L, 20);
        return;
    }

    // Pre-smooth
    smooth(L, 2);

    // Restrict residual to coarse
    AggLevel& coarse = levels_[level + 1];
    restrictResidual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);

    // Recurse
    vCycle(level + 1, nlevels);

    // Prolongate correction
    prolongateAdd(coarse, L);

    // Post-smooth
    smooth(L, 2);
}

// ── Apply preconditioner: z = M^{-1} * r ──
void AMGPreconditioner::apply(const Grid& g, const std::vector<double>& r,
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

    // Build aggregates between each adjacent level pair
    for (int l = 0; l < nl - 1; l++)
        buildAggregates(levels_[l], levels_[l+1]);

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

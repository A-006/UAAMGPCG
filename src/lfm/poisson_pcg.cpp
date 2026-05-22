#include "lfm/poisson_pcg.h"
#include <algorithm>
#include <cmath>
#include <vector>

// ── Coarse-level descriptor ──
struct Level {
    int nx, ny;
    double dx, dy;
    std::vector<double> p;   // pressure
    std::vector<double> b;   // RHS
    std::vector<bool> solid; // solid mask
};

// ── Build multi-level hierarchy ──
static std::vector<Level> build_levels(const Grid& fine) {
    std::vector<Level> lv;
    int nx = fine.nx, ny = fine.ny;
    double dx = fine.dx, dy = fine.dy;

    while (nx >= 2 && ny >= 2) {
        Level L;
        L.nx = nx; L.ny = ny;
        L.dx = dx; L.dy = dy;
        L.p.resize((nx+2)*(ny+2), 0.0);
        L.b.resize((nx+2)*(ny+2), 0.0);
        L.solid.resize((nx+2)*(ny+2), false);
        lv.push_back(std::move(L));

        if (nx <= 4 || ny <= 4) break;
        nx = nx / 2;
        ny = ny / 2;
        dx *= 2.0;
        dy *= 2.0;
    }
    return lv;
}

// ── Restrict solid mask from fine to coarse ──
static void restrict_solid(const Level& fine, Level& coarse) {
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

// ── RBGS smoothing at a given level ──
static void level_rbgs(Level& L, int sweeps) {
    double idx2 = 1.0/(L.dx*L.dx), idy2 = 1.0/(L.dy*L.dy);
    double diag = 2.0*(idx2+idy2);
    int stride = L.ny + 2;

    for (int s = 0; s < sweeps; s++) {
        // Red
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
        // Black
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

// ── Restrict residual: fine → coarse (4-to-1 average) ──
static void restrict_residual(const Level& fine, Level& coarse) {
    int fs = fine.ny + 2, cs = coarse.ny + 2;
    // Compute residual on fine, restrict to coarse
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

// ── Prolongate correction: coarse → fine (constant interpolation) ──
static void prolongate_add(const Level& coarse, Level& fine) {
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

// ── Recursive V-Cycle ──
static void v_cycle(std::vector<Level>& lv, int level, int nlevels) {
    Level& L = lv[level];
    int stride = L.ny + 2;

    if (level == nlevels - 1) {
        level_rbgs(L, 20);
        return;
    }

    // Pre-smooth
    level_rbgs(L, 2);

    // Restrict residual to coarse
    Level& coarse = lv[level + 1];
    restrict_residual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);

    // Recurse
    v_cycle(lv, level + 1, nlevels);

    // Prolongate correction
    prolongate_add(coarse, L);

    // Post-smooth
    level_rbgs(L, 2);
}

// ── PCG: CG with V-Cycle preconditioner ──
int pcg_solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol) {
    const int nx = g.nx, ny = g.ny;

    // Zero-mean RHS
    std::vector<double> rhs = rhs_in;
    {
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) { sum += rhs[g.ip(i,j)]; count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) rhs[g.ip(i,j)] -= mean;
    }

    // Build V-Cycle hierarchy
    auto levels = build_levels(g);
    int nl = (int)levels.size();

    // Copy solid mask to finest level
    for (int i = 0; i <= nx+1; i++)
        for (int j = 0; j <= ny+1; j++)
            levels[0].solid[i*(ny+2)+j] = g.is_solid(i,j);

    // Propagate solid masks to coarser levels
    for (int l = 1; l < nl; l++)
        restrict_solid(levels[l-1], levels[l]);

    // A*x operator on finest level
    auto matvec = [&](const std::vector<double>& v, std::vector<double>& Av) {
        double idx2 = 1.0/(g.dx*g.dx), idy2 = 1.0/(g.dy*g.dy);
        double diag = 2.0*(idx2+idy2);
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++) {
                int idx = g.ip(i,j);
                if (g.is_solid(i,j)) { Av[idx] = 0.0; continue; }
                double vL = (i>1 && !g.is_solid(i-1,j)) ? v[g.ip(i-1,j)] : v[idx];
                double vR = (i<nx && !g.is_solid(i+1,j)) ? v[g.ip(i+1,j)] : v[idx];
                double vB = (j>1 && !g.is_solid(i,j-1)) ? v[g.ip(i,j-1)] : v[idx];
                double vT = (j<ny && !g.is_solid(i,j+1)) ? v[g.ip(i,j+1)] : v[idx];
                Av[idx] = diag * v[idx] - (vL+vR)*idx2 - (vB+vT)*idy2;
            }
    };

    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) s += a[g.ip(i,j)] * b[g.ip(i,j)];
        return s;
    };

    auto subtract_mean = [&](std::vector<double>& v) {
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) { sum += v[g.ip(i,j)]; count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) v[g.ip(i,j)] -= mean;
    };

    // r = b - Ax (x=0)
    std::vector<double> r = rhs;
    subtract_mean(r);

    // Preconditioner: z = M^{-1} * r  (one V-Cycle)
    auto precondition = [&](std::vector<double>& z, const std::vector<double>& r_src) {
        // Copy r to finest level b
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                levels[0].b[g.ip(i,j)] = r_src[g.ip(i,j)];
        // Zero all level pressures
        for (int l = 0; l < nl; l++)
            std::fill(levels[l].p.begin(), levels[l].p.end(), 0.0);
        // One V-Cycle
        v_cycle(levels, 0, nl);
        // Copy finest level p to z
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                z[g.ip(i,j)] = levels[0].p[g.ip(i,j)];
    };

    std::vector<double> z(rhs.size());
    precondition(z, r);

    std::vector<double> p = z;
    double rsold = dot(r, z);

    std::vector<double> Ap(rhs.size());

    for (int k = 0; k < max_iter; k++) {
        matvec(p, Ap);

        double pAp = dot(p, Ap);
        if (pAp < 1e-15) return k + 1;

        double alpha = rsold / pAp;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) {
                    int idx = g.ip(i,j);
                    g.p[idx] += alpha * p[idx];
                    r[idx]   -= alpha * Ap[idx];
                }

        subtract_mean(r);

        double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < tol) return k + 1;

        precondition(z, r);

        double beta = dot(r, z) / rsold;  // Use (r_new, z_new) / (r_old, z_old)
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) {
                    int idx = g.ip(i,j);
                    p[idx] = z[idx] + beta * p[idx];
                }
        rsold = dot(r, z);
    }

    return max_iter;
}

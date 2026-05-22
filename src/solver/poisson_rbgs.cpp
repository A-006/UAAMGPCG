#include "solver/poisson_rbgs.h"
#include <algorithm>

void RBGSSolver::solve(Grid& g, const std::vector<double>& rhs_in,
                       int max_iter, double /*tol*/) {
    const double idx2 = 1.0 / (g.dx * g.dx);
    const double idy2 = 1.0 / (g.dy * g.dy);
    const double diag = 2.0 * (idx2 + idy2);
    const int    nx = g.nx, ny = g.ny;

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

    for (int iter = 0; iter < max_iter; iter++) {
        // Red points
        for (int i = 1; i <= nx; i++)
            for (int j = 1 + (i % 2); j <= ny; j += 2) {
                if (g.is_solid(i,j)) continue;
                double pL = (i > 1 && !g.is_solid(i-1,j)) ? g.p_at(i-1,j) : g.p_at(i,j);
                double pR = (i < nx && !g.is_solid(i+1,j)) ? g.p_at(i+1,j) : g.p_at(i,j);
                double pB = (j > 1 && !g.is_solid(i,j-1)) ? g.p_at(i,j-1) : g.p_at(i,j);
                double pT = (j < ny && !g.is_solid(i,j+1)) ? g.p_at(i,j+1) : g.p_at(i,j);
                double lap = (pL + pR) * idx2 + (pB + pT) * idy2;
                double eff_d = diag;
                if (i == 1  || g.is_solid(i-1,j)) eff_d -= idx2;
                if (i == nx || g.is_solid(i+1,j)) eff_d -= idx2;
                if (j == 1  || g.is_solid(i,j-1)) eff_d -= idy2;
                if (j == ny || g.is_solid(i,j+1)) eff_d -= idy2;
                double inv_d = (eff_d < 1e-15) ? 0.0 : 1.0 / eff_d;
                g.p[g.ip(i,j)] += inv_d * (rhs[g.ip(i,j)] - lap);
            }
        // Black points
        for (int i = 1; i <= nx; i++)
            for (int j = 1 + ((i+1) % 2); j <= ny; j += 2) {
                if (g.is_solid(i,j)) continue;
                double pL = (i > 1 && !g.is_solid(i-1,j)) ? g.p_at(i-1,j) : g.p_at(i,j);
                double pR = (i < nx && !g.is_solid(i+1,j)) ? g.p_at(i+1,j) : g.p_at(i,j);
                double pB = (j > 1 && !g.is_solid(i,j-1)) ? g.p_at(i,j-1) : g.p_at(i,j);
                double pT = (j < ny && !g.is_solid(i,j+1)) ? g.p_at(i,j+1) : g.p_at(i,j);
                double lap = (pL + pR) * idx2 + (pB + pT) * idy2;
                double eff_d = diag;
                if (i == 1  || g.is_solid(i-1,j)) eff_d -= idx2;
                if (i == nx || g.is_solid(i+1,j)) eff_d -= idx2;
                if (j == 1  || g.is_solid(i,j-1)) eff_d -= idy2;
                if (j == ny || g.is_solid(i,j+1)) eff_d -= idy2;
                double inv_d = (eff_d < 1e-15) ? 0.0 : 1.0 / eff_d;
                g.p[g.ip(i,j)] += inv_d * (rhs[g.ip(i,j)] - lap);
            }
        // Remove null space
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) { sum += g.p_at(i,j); count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) g.p_at(i,j) -= mean;
    }
}

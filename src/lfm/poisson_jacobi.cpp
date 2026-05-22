#include "lfm/poisson_jacobi.h"
#include <algorithm>

void jacobi_solve(Grid& g, const std::vector<double>& rhs, int max_iter) {
    const double idx2 = 1.0 / (g.dx * g.dx);
    const double idy2 = 1.0 / (g.dy * g.dy);
    const double diag = 2.0 * (idx2 + idy2);
    const int    nx = g.nx, ny = g.ny;

    std::vector<double> pn(g.p.size());

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 1; i <= nx; i++) {
            for (int j = 1; j <= ny; j++) {
                if (g.is_solid(i,j)) {
                    pn[g.ip(i,j)] = 0.0;
                    continue;
                }

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

                if (eff_d < 1e-15)
                    pn[g.ip(i,j)] = 0.0;
                else
                    pn[g.ip(i,j)] = (lap - rhs[g.ip(i,j)]) / eff_d;
            }
        }
        std::copy(pn.begin(), pn.end(), g.p.begin());
    }
}

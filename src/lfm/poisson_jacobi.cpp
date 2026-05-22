#include "lfm/poisson_jacobi.h"
#include <algorithm>

void jacobi_solve(Grid& g, const std::vector<double>& rhs_in, int max_iter) {
    const double idx2 = 1.0 / (g.dx * g.dx);
    const double idy2 = 1.0 / (g.dy * g.dy);
    const double diag = 2.0 * (idx2 + idy2);
    const int    nx = g.nx, ny = g.ny;
    const int    stride = nx + 2;

    // Neumann BC: subtract RHS mean so the system is compatible
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

    std::vector<double> p_old = g.p;

    for (int iter = 0; iter < max_iter; iter++) {
        // Jacobi sweep: compute all new values from p_old, write to g.p
        for (int i = 1; i <= nx; i++) {
            for (int j = 1; j <= ny; j++) {
                if (g.is_solid(i,j)) continue;

                // Ax from OLD pressure p_old (Jacobi — all points use old values)
                double pL = (i > 1 && !g.is_solid(i-1,j))
                    ? p_old[g.ip(i-1,j)] : p_old[g.ip(i,j)];
                double pR = (i < nx && !g.is_solid(i+1,j))
                    ? p_old[g.ip(i+1,j)] : p_old[g.ip(i,j)];
                double pB = (j > 1 && !g.is_solid(i,j-1))
                    ? p_old[g.ip(i,j-1)] : p_old[g.ip(i,j)];
                double pT = (j < ny && !g.is_solid(i,j+1))
                    ? p_old[g.ip(i,j+1)] : p_old[g.ip(i,j)];
                double lap = (pL + pR) * idx2 + (pB + pT) * idy2;

                // Effective diagonal
                double eff_d = diag;
                if (i == 1  || g.is_solid(i-1,j)) eff_d -= idx2;
                if (i == nx || g.is_solid(i+1,j)) eff_d -= idx2;
                if (j == 1  || g.is_solid(i,j-1)) eff_d -= idy2;
                if (j == ny || g.is_solid(i,j+1)) eff_d -= idy2;
                double inv_d = (eff_d < 1e-15) ? 0.0 : 1.0 / eff_d;

                int idx = g.ip(i,j);
                g.p[idx] = p_old[idx] + inv_d * (rhs[idx] - lap);
            }
        }

        // Remove null space (Neumann BC: solution unique only up to constant)
        {
            double sum = 0; int count = 0;
            for (int i = 1; i <= nx; i++)
                for (int j = 1; j <= ny; j++)
                    if (!g.is_solid(i,j)) { sum += g.p_at(i,j); count++; }
            double mean = (count > 0) ? sum / count : 0.0;
            for (int i = 1; i <= nx; i++)
                for (int j = 1; j <= ny; j++)
                    if (!g.is_solid(i,j)) g.p_at(i,j) -= mean;
        }

        p_old = g.p;
    }
}

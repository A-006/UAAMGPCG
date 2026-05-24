/**
 * @file jacobi_3d.cpp
 * @brief Jacobi solver for 3D — damped diagonal preconditioned Richardson iteration.
 * @author liutao
 * @date 2026-05-24
 */
#include "solver/jacobi_3d.h"
#include <algorithm>

void Jacobi3D::solve(Grid3D& g, const std::vector<double>& rhs_in,
                     int max_iter, double /*tol*/) {
    const double idx2 = 1.0 / (g.dx * g.dx);
    const double idy2 = 1.0 / (g.dy * g.dy);
    const double idz2 = 1.0 / (g.dz * g.dz);
    const double diag = 2.0 * (idx2 + idy2 + idz2);
    const int    nx = g.nx, ny = g.ny, nz = g.nz;

    std::vector<double> rhs = rhs_in;
    {
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) { sum += rhs[g.ip(i,j,k)]; count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) rhs[g.ip(i,j,k)] -= mean;
    }

    std::vector<double> p_old = g.p;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 1; i <= nx; i++) {
            for (int j = 1; j <= ny; j++) {
                for (int k = 1; k <= nz; k++) {
                    if (g.is_solid(i,j,k)) continue;

                    double pL = (i > 1 && !g.is_solid(i-1,j,k))
                        ? p_old[g.ip(i-1,j,k)] : p_old[g.ip(i,j,k)];
                    double pR = (i < nx && !g.is_solid(i+1,j,k))
                        ? p_old[g.ip(i+1,j,k)] : p_old[g.ip(i,j,k)];
                    double pB = (j > 1 && !g.is_solid(i,j-1,k))
                        ? p_old[g.ip(i,j-1,k)] : p_old[g.ip(i,j,k)];
                    double pT = (j < ny && !g.is_solid(i,j+1,k))
                        ? p_old[g.ip(i,j+1,k)] : p_old[g.ip(i,j,k)];
                    double pF = (k > 1 && !g.is_solid(i,j,k-1))
                        ? p_old[g.ip(i,j,k-1)] : p_old[g.ip(i,j,k)];
                    double pK = (k < nz && !g.is_solid(i,j,k+1))
                        ? p_old[g.ip(i,j,k+1)] : p_old[g.ip(i,j,k)];
                    double lap = (pL + pR) * idx2 + (pB + pT) * idy2
                               + (pF + pK) * idz2;

                    double eff_d = diag;
                    if (i == 1  || g.is_solid(i-1,j,k)) eff_d -= idx2;
                    if (i == nx || g.is_solid(i+1,j,k)) eff_d -= idx2;
                    if (j == 1  || g.is_solid(i,j-1,k)) eff_d -= idy2;
                    if (j == ny || g.is_solid(i,j+1,k)) eff_d -= idy2;
                    if (k == 1  || g.is_solid(i,j,k-1)) eff_d -= idz2;
                    if (k == nz || g.is_solid(i,j,k+1)) eff_d -= idz2;
                    double inv_d = (eff_d < 1e-15) ? 0.0 : 1.0 / eff_d;

                    g.p[g.ip(i,j,k)] = p_old[g.ip(i,j,k)]
                                     + inv_d * (rhs[g.ip(i,j,k)] - lap);
                }
            }
        }

        // Remove null space
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) { sum += g.p_at(i,j,k); count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) g.p_at(i,j,k) -= mean;

        p_old = g.p;
    }
}

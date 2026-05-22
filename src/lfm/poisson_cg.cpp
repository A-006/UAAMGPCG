#include "lfm/poisson_cg.h"
#include <algorithm>
#include <cmath>

// Matrix-vector product: y = A * x  on MAC grid
static void mac_matvec(const Grid& g, const std::vector<double>& x,
                       std::vector<double>& y) {
    const double idx2 = 1.0 / (g.dx * g.dx);
    const double idy2 = 1.0 / (g.dy * g.dy);
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j)) { y[g.ip(i,j)] = 0.0; continue; }
            double vL = (i > 1 && !g.is_solid(i-1,j)) ? x[g.ip(i-1,j)] : x[g.ip(i,j)];
            double vR = (i < g.nx && !g.is_solid(i+1,j)) ? x[g.ip(i+1,j)] : x[g.ip(i,j)];
            double vB = (j > 1 && !g.is_solid(i,j-1)) ? x[g.ip(i,j-1)] : x[g.ip(i,j)];
            double vT = (j < g.ny && !g.is_solid(i,j+1)) ? x[g.ip(i,j+1)] : x[g.ip(i,j)];
            // Ax =  -Laplacian = -( (vL+vR-2v)/dx² + (vB+vT-2v)/dy² )
            //    =  (2/dx²+2/dy²)*v - (vL+vR)/dx² - (vB+vT)/dy²
            double diag = 2.0*(idx2+idy2);
            y[g.ip(i,j)] = diag * x[g.ip(i,j)] - (vL+vR)*idx2 - (vB+vT)*idy2;
        }
    }
}

// Dot product over non-solid interior cells
static double mac_dot(const Grid& g, const std::vector<double>& a,
                      const std::vector<double>& b) {
    double s = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i,j))
                s += a[g.ip(i,j)] * b[g.ip(i,j)];
    return s;
}

// y += a * x
static void mac_axpy(double a, const std::vector<double>& x,
                     std::vector<double>& y, const Grid& g) {
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i,j))
                y[g.ip(i,j)] += a * x[g.ip(i,j)];
}

int cg_solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol) {
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

    // r = b - A*x  (x=0 initially, so r=b)
    std::vector<double> r = rhs;
    // Remove mean from initial residual
    {
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) { sum += r[g.ip(i,j)]; count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) r[g.ip(i,j)] -= mean;
    }

    std::vector<double> p = r;    // p = r
    double rsold = mac_dot(g, r, r);

    std::vector<double> Ap(rhs.size());
    std::vector<double>& x = g.p;

    for (int k = 0; k < max_iter; k++) {
        mac_matvec(g, p, Ap);
        double pAp = mac_dot(g, p, Ap);
        if (pAp < 1e-15) return k + 1;

        double alpha = rsold / pAp;
        mac_axpy( alpha, p, x, g);    // x += alpha*p
        mac_axpy(-alpha, Ap, r, g);   // r -= alpha*Ap

        // Remove mean from residual (Neumann null space)
        {
            double sum = 0; int count = 0;
            for (int i = 1; i <= nx; i++)
                for (int j = 1; j <= ny; j++)
                    if (!g.is_solid(i,j)) { sum += r[g.ip(i,j)]; count++; }
            double mean = (count > 0) ? sum / count : 0.0;
            for (int i = 1; i <= nx; i++)
                for (int j = 1; j <= ny; j++)
                    if (!g.is_solid(i,j)) r[g.ip(i,j)] -= mean;
        }

        double rsnew = mac_dot(g, r, r);
        if (std::sqrt(rsnew) < tol) return k + 1;

        double beta = rsnew / rsold;
        // p = r + beta*p
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) {
                    int idx = g.ip(i,j);
                    p[idx] = r[idx] + beta * p[idx];
                }

        rsold = rsnew;
    }

    // Remove mean from solution
    {
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) { sum += x[g.ip(i,j)]; count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) x[g.ip(i,j)] -= mean;
    }

    return max_iter;
}

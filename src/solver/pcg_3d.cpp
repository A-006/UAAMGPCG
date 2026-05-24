/**
 * @file pcg_3d.cpp
 * @brief Preconditioned Conjugate Gradient for 3D — CG accelerated by a pluggable preconditioner.
 * @author liutao
 * @date 2026-05-24
 */
#include "solver/pcg_3d.h"
#include <algorithm>
#include <cmath>

PCG3D::PCG3D(std::unique_ptr<Preconditioner3D> p)
    : precond_(std::move(p)) {}

std::string PCG3D::name() const {
    return "PCG3D(" + precond_->name() + ")";
}

void PCG3D::solve(Grid3D& g, const std::vector<double>& rhs_in,
                  int max_iter, double tol) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;

    // Zero-mean RHS, then negate because matvec is -nabla^2
    // PCG solves (-nabla^2) p = -rhs_in  =>  nabla^2 p = rhs_in
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
                    if (!g.is_solid(i,j,k)) rhs[g.ip(i,j,k)] = -(rhs[g.ip(i,j,k)] - mean);
    }

    auto matvec = [&](const std::vector<double>& v, std::vector<double>& Av) {
        double idx2 = 1.0/(g.dx*g.dx), idy2 = 1.0/(g.dy*g.dy), idz2 = 1.0/(g.dz*g.dz);
        double diag = 2.0*(idx2+idy2+idz2);
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++) {
                    int idx = g.ip(i,j,k);
                    if (g.is_solid(i,j,k)) { Av[idx] = 0.0; continue; }
                    double vL = (i>1 && !g.is_solid(i-1,j,k)) ? v[g.ip(i-1,j,k)] : v[idx];
                    double vR = (i<nx && !g.is_solid(i+1,j,k)) ? v[g.ip(i+1,j,k)] : v[idx];
                    double vB = (j>1 && !g.is_solid(i,j-1,k)) ? v[g.ip(i,j-1,k)] : v[idx];
                    double vT = (j<ny && !g.is_solid(i,j+1,k)) ? v[g.ip(i,j+1,k)] : v[idx];
                    double vF = (k>1 && !g.is_solid(i,j,k-1)) ? v[g.ip(i,j,k-1)] : v[idx];
                    double vK = (k<nz && !g.is_solid(i,j,k+1)) ? v[g.ip(i,j,k+1)] : v[idx];
                    Av[idx] = diag * v[idx]
                            - (vL+vR)*idx2 - (vB+vT)*idy2 - (vF+vK)*idz2;
                }
    };

    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) s += a[g.ip(i,j,k)] * b[g.ip(i,j,k)];
        return s;
    };

    auto subtract_mean = [&](std::vector<double>& v) {
        double sum = 0; int count = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) { sum += v[g.ip(i,j,k)]; count++; }
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) v[g.ip(i,j,k)] -= mean;
    };

    // r = b - A*x  (x=0 initially)
    std::vector<double> r = rhs;
    subtract_mean(r);

    // z = M^{-1} * r
    std::vector<double> z(rhs.size());
    precond_->apply(g, r, z);
    subtract_mean(z);

    std::vector<double> p = z;
    double rsold = dot(r, z);

    std::vector<double> Ap(rhs.size());

    for (int k = 0; k < max_iter; k++) {
        matvec(p, Ap);

        double pAp = dot(p, Ap);
        if (pAp < 1e-15) return;

        double alpha = rsold / pAp;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) {
                        int idx = g.ip(i,j,k);
                        g.p[idx] += alpha * p[idx];
                        r[idx]   -= alpha * Ap[idx];
                    }

        double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < tol) return;

        precond_->apply(g, r, z);
        subtract_mean(z);

        double beta = dot(r, z) / rsold;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    if (!g.is_solid(i,j,k)) {
                        int idx = g.ip(i,j,k);
                        p[idx] = z[idx] + beta * p[idx];
                    }
        rsold = dot(r, z);
    }
}

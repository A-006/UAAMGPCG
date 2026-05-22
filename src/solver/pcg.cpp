/**
 * @file pcg.cpp
 * @brief Preconditioned Conjugate Gradient — CG accelerated by a pluggable preconditioner.
 * @author liutao
 * @date 2026-05-22
 */
#include "solver/pcg.h"
#include <algorithm>
#include <cmath>

PCG::PCG(std::unique_ptr<Preconditioner> p)
    : precond_(std::move(p)) {}

std::string PCG::name() const {
    return "PCG(" + precond_->name() + ")";
}

void PCG::solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol) {
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

    // r = b - A*x  (x=0 initially)
    std::vector<double> r = rhs;
    subtract_mean(r);

    // z = M^{-1} * r
    std::vector<double> z(rhs.size());
    precond_->apply(g, r, z);

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
                if (!g.is_solid(i,j)) {
                    int idx = g.ip(i,j);
                    g.p[idx] += alpha * p[idx];
                    r[idx]   -= alpha * Ap[idx];
                }

        subtract_mean(r);

        double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < tol) return;

        precond_->apply(g, r, z);

        double beta = dot(r, z) / rsold;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                if (!g.is_solid(i,j)) {
                    int idx = g.ip(i,j);
                    p[idx] = z[idx] + beta * p[idx];
                }
        rsold = dot(r, z);
    }
}

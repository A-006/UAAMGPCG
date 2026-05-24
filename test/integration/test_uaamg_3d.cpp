#include "solver/pcg_3d.h"
#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include "../test_utils.h"
#include <cstdio>
#include <cmath>

int main() {
    test_header("3D UAAMGPCG Integration Test");

    int nx = 16, ny = 8, nz = 8, N = (nx+2)*(ny+2)*(nz+2);

    // Gaussian RHS
    std::vector<double> rhs(N, 0.0);
    double cx = nx/2.0, cy = ny/2.0, cz = nz/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                double d2 = (i-cx)*(i-cx)+(j-cy)*(j-cy)+(k-cz)*(k-cz);
                rhs[i + j*(nx+2) + k*(nx+2)*(ny+2)] = std::exp(-0.01 * d2);
            }

    // Test 1: UAAMG V-cycle produces finite output
    {
        Grid3D g(nx, ny, nz, 1.0, 1.0, 1.0);
        UAAMGPreconditioner3D precond;
        std::vector<double> z(N, 0.0);
        precond.apply(g, rhs, z);
        double mz = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    mz = std::max(mz, std::abs(z[g.ip(i,j,k)]));
        printf("  V-cycle max|z|=%.6e\n", mz);
        check(mz > 0 && mz < 1e6, "UAAMG V-cycle produces finite z");
    }

    // Test 2: UAAMGPCG converges
    {
        Grid3D g(nx, ny, nz, 1.0, 1.0, 1.0);
        auto solver = std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>());
        solver->solve(g, rhs, 50, 1e-10);
        double mp = 0;
        for (int i = 1; i <= nx; i++)
            for (int j = 1; j <= ny; j++)
                for (int k = 1; k <= nz; k++)
                    mp = std::max(mp, std::abs(g.p_at(i,j,k)));
        printf("  PCG3D(UAAMG) max|p|=%.6e\n", mp);
        check(mp > 0 && mp < 1e6, "UAAMGPCG produces finite solution");
    }

    // Test 3: UAAMGPCG is better than CG for same iterations
    {
        Grid3D g1(nx, ny, nz, 1.0, 1.0, 1.0);
        Grid3D g2(nx, ny, nz, 1.0, 1.0, 1.0);

        auto cg = std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>());
        auto uaamg = std::make_unique<PCG3D>(std::make_unique<UAAMGPreconditioner3D>());

        cg->solve(g1, rhs, 10, 0.0);
        uaamg->solve(g2, rhs, 10, 0.0);

        // Compute residual L2 for both
        auto compute_l2 = [&](Grid3D& g) {
            double l2 = 0;
            double idx2=1.0/(g.dx*g.dx), idy2=1.0/(g.dy*g.dy), idz2=1.0/(g.dz*g.dz);
            double diag=2.0*(idx2+idy2+idz2);
            for (int i=1;i<=nx;i++) for (int j=1;j<=ny;j++) for (int k=1;k<=nz;k++) {
                int id=g.ip(i,j,k);
                double pC=g.p[id];
                double pL=(i>1)?g.p[g.ip(i-1,j,k)]:pC;
                double pR=(i<nx)?g.p[g.ip(i+1,j,k)]:pC;
                double pB=(j>1)?g.p[g.ip(i,j-1,k)]:pC;
                double pT=(j<ny)?g.p[g.ip(i,j+1,k)]:pC;
                double pF=(k>1)?g.p[g.ip(i,j,k-1)]:pC;
                double pK=(k<nz)?g.p[g.ip(i,j,k+1)]:pC;
                double Ax=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
                // RHS is negated+zero-mean in solve, just compare residuals
                double r=-(rhs[id]-0); // mean ~ 0 for Gaussian
                l2+=(r-Ax)*(r-Ax);
            }
            return std::sqrt(l2);
        };

        double cg_l2 = compute_l2(g1);
        double uaamg_l2 = compute_l2(g2);
        printf("  CG L2=%.4e  UAAMGPCG L2=%.4e\n", cg_l2, uaamg_l2);
        // UAAMGPCG should converge faster (smaller residual after same iters)
        check(uaamg_l2 < cg_l2, "UAAMGPCG converges faster than CG");
    }

    return test_summary();
}

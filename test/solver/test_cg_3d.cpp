#include "solver/pcg_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    int nx = 16, ny = 8, nz = 8, N = (nx+2)*(ny+2)*(nz+2);
    Grid3D g(nx, ny, nz, 1.0, 1.0, 1.0);

    std::vector<double> rhs(N, 0.0);
    double cx = nx/2.0, cy = ny/2.0, cz = nz/2.0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++) {
                double d2 = (i-cx)*(i-cx)+(j-cy)*(j-cy)+(k-cz)*(k-cz);
                rhs[g.ip(i,j,k)] = std::exp(-0.01 * d2);
            }

    auto solver = std::make_unique<PCG3D>(std::make_unique<IdentityPreconditioner3D>());
    solver->solve(g, rhs, 200, 1e-10);

    double max_p = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                max_p = std::max(max_p, std::abs(g.p_at(i,j,k)));

    printf("PCG3D(CG) 200 iter: max|p|=%e\n", max_p);
    bool pass = max_p > 1e-15 && max_p < 1.0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

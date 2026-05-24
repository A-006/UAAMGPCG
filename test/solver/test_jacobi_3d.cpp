#include "solver/jacobi_3d.h"
#include <cstdio>
#include <cmath>

int main() {
    int nx = 16, ny = 8, nz = 8, N = (nx+2)*(ny+2)*(nz+2);
    Grid3D g(nx, ny, nz, 1.0, 1.0, 1.0);

    std::vector<double> rhs(N, 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                rhs[g.ip(i,j,k)] = 1.0;

    Jacobi3D solver;
    solver.solve(g, rhs, 100, 0.0);

    double max_p = 0;
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            for (int k = 1; k <= nz; k++)
                max_p = std::max(max_p, std::abs(g.p_at(i,j,k)));

    printf("Jacobi3D 100 sweeps: max|p|=%e\n", max_p);
    bool pass = max_p < 1.0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

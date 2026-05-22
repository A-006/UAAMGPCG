#include "core/grid.h"
#include "advection/advection.h"
#include "boundary/boundary.h"
#include "solver/poisson_jacobi.h"
#include "test_utils.h"
#include <cmath>

int main() {
    test_header("LFM 2D Fluid Simulation Unit Tests");

    // Test 1: Grid construction and dimensions
    {
        Grid g(32, 16, 4.0, 1.0);
        check(g.nx == 32 && g.ny == 16, "Grid dimensions nx=32, ny=16");
        check(g.dx == 0.125, "dx = Lx/nx = 0.125");
        check(g.dy == 0.0625, "dy = Ly/ny = 0.0625");
        int u_size = (32+1)*(16+2);
        check((int)g.u.size() == u_size, "u array size = (nx+1)*(ny+2)");
        int v_size = (32+2)*(16+1);
        check((int)g.v.size() == v_size, "v array size = (nx+2)*(ny+1)");
    }

    // Test 2: Divergence of uniform flow is zero
    {
        Grid g(16, 8, 1.0, 1.0);
        for (int i = 0; i <= 16; i++)
            for (int j = 1; j <= 8; j++)
                g.u_at(i,j) = 1.0;
        for (int i = 1; i <= 16; i++)
            for (int j = 0; j <= 8; j++)
                g.v_at(i,j) = 0.0;
        double max_div = 0;
        for (int i = 1; i <= 16; i++)
            for (int j = 1; j <= 8; j++)
                max_div = std::max(max_div, std::abs(g.divergence(i,j)));
        check(max_div < 1e-12, "uniform flow divergence = 0");
    }

    // Test 3: u = x, v = -y gives zero divergence
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int i = 0; i <= 8; i++)
            for (int j = 1; j <= 8; j++)
                g.u_at(i,j) = i * g.dx;
        for (int i = 1; i <= 8; i++)
            for (int j = 0; j <= 8; j++)
                g.v_at(i,j) = -(j * g.dy);
        double max_div = 0;
        for (int i = 1; i <= 8; i++)
            for (int j = 1; j <= 8; j++)
                max_div = std::max(max_div, std::abs(g.divergence(i,j)));
        check(max_div < 1e-12, "u=x, v=-y => div = 0");
    }

    // Test 4: sampleU is exact at u-face centers
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int i = 0; i <= 8; i++)
            for (int j = 1; j <= 8; j++)
                g.u_at(i,j) = i * 10.0 + j;
        bool ok = true;
        for (int i = 0; i <= 8; i++)
            for (int j = 1; j <= 8; j++) {
                double x = i * g.dx, y = (j - 0.5) * g.dy;
                if (std::abs(AdvectionScheme::sampleU(g,x,y) - g.u_at(i,j)) > 1e-12) ok = false;
            }
        check(ok, "sampleU exact at u-face centers");
    }

    // Test 5: sampleV is exact at v-face centers
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int i = 1; i <= 8; i++)
            for (int j = 0; j <= 8; j++)
                g.v_at(i,j) = i * 10.0 + j;
        bool ok = true;
        for (int i = 1; i <= 8; i++)
            for (int j = 0; j <= 8; j++) {
                double x = (i - 0.5) * g.dx, y = j * g.dy;
                if (std::abs(AdvectionScheme::sampleV(g,x,y) - g.v_at(i,j)) > 1e-12) ok = false;
            }
        check(ok, "sampleV exact at v-face centers");
    }

    // Test 6: clamp
    {
        check(Grid::clamp(-1.0, 0.0, 10.0) == 0.0, "clamp lower bound");
        check(Grid::clamp(100.0, 0.0, 10.0) == 10.0, "clamp upper bound");
        check(Grid::clamp(5.0, 0.0, 10.0) == 5.0, "clamp in range");
    }

    // Test 7: Pressure array indexing is consistent
    {
        Grid g(16, 8, 1.0, 1.0);
        g.p_at(4, 3) = 5.0;
        check(g.p_at(4, 3) == 5.0, "pressure array write/read consistent");
        check(g.p_at(4, 4) == 0.0, "adjacent pressure cell independent");
    }

    // Test 8: Solid domain marking
    {
        Grid g(16, 8, 1.0, 1.0);
        check(!g.is_solid(4,4), "default cell not solid");
        g.set_solid(4,4);
        check(g.is_solid(4,4), "set_solid marks cell as solid");
        check(!g.is_solid(5,5), "adjacent cell unaffected");
    }

    // Test 9: Jacobi Poisson solver with manufactured solution
    // p_exact = sin(2*pi*x)*sin(2*pi*y)  (zero-mean, compatible with Neumann BC)
    {
        Grid g(32, 32, 1.0, 1.0);
        std::vector<double> rhs(g.p.size(), 0.0);
        for (int i = 1; i <= 32; i++)
            for (int j = 1; j <= 32; j++) {
                double x = (i-0.5)*g.dx, y = (j-0.5)*g.dy;
                rhs[g.ip(i,j)] = 8.0*M_PI*M_PI * std::sin(2*M_PI*x) * std::sin(2*M_PI*y);
            }

        std::fill(g.p.begin(), g.p.end(), 0.0);
        JacobiSolver solver;
        solver.solve(g, rhs, 5000, 0.0);

        double max_res = 0;
        for (int i = 2; i <= 31; i++)
            for (int j = 2; j <= 31; j++) {
                double lap = (g.p_at(i+1,j) + g.p_at(i-1,j) - 2*g.p_at(i,j))/(g.dx*g.dx)
                           + (g.p_at(i,j+1) + g.p_at(i,j-1) - 2*g.p_at(i,j))/(g.dy*g.dy);
                max_res = std::max(max_res, std::abs(lap - rhs[g.ip(i,j)]));
            }
        check(max_res < 1.0, "Jacobi Poisson solve converges (manufactured solution)");
    }

    return test_summary();
}

// Unit tests for LFM 2D fluid simulation — exercises code in ../src/lfm_2d.cpp
#define TEST_MODE
#include "../src/lfm_2d.cpp"
#include "test_utils.h"

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
                max_div = std::max(max_div, std::abs(divergence(g,i,j)));
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
                max_div = std::max(max_div, std::abs(divergence(g,i,j)));
        check(max_div < 1e-12, "u=x, v=-y => div = 0");
    }

    // Test 4: sample_u is exact at u-face centers
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int i = 0; i <= 8; i++)
            for (int j = 1; j <= 8; j++)
                g.u_at(i,j) = i * 10.0 + j;
        bool ok = true;
        for (int i = 0; i <= 8; i++)
            for (int j = 1; j <= 8; j++) {
                double x = i * g.dx, y = (j - 0.5) * g.dy;
                if (std::abs(sample_u(g,x,y) - g.u_at(i,j)) > 1e-12) ok = false;
            }
        check(ok, "sample_u exact at u-face centers");
    }

    // Test 5: sample_v is exact at v-face centers
    {
        Grid g(8, 8, 1.0, 1.0);
        for (int i = 1; i <= 8; i++)
            for (int j = 0; j <= 8; j++)
                g.v_at(i,j) = i * 10.0 + j;
        bool ok = true;
        for (int i = 1; i <= 8; i++)
            for (int j = 0; j <= 8; j++) {
                double x = (i - 0.5) * g.dx, y = j * g.dy;
                if (std::abs(sample_v(g,x,y) - g.v_at(i,j)) > 1e-12) ok = false;
            }
        check(ok, "sample_v exact at v-face centers");
    }

    // Test 6: clamps out-of-bounds coordinates
    {
        check(clamp(-1.0, 0.0, 10.0) == 0.0, "clamp lower bound");
        check(clamp(100.0, 0.0, 10.0) == 10.0, "clamp upper bound");
        check(clamp(5.0, 0.0, 10.0) == 5.0, "clamp in range");
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

    return test_summary();
}

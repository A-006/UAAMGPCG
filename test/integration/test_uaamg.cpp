/**
 * @file test_uaamg.cpp
 * @brief Unit tests for the paper's UAAMGPCG solver.
 *
 * Tests:
 *   1. UAAMG V-cycle reduces residual (standalone)
 *   2. PCG/UAAMG converges on Poisson problem
 *   3. PCG/UAAMG passes Karman divergence check
 *   4. Comparison with CG
 * @author liutao
 * @date 2026-05-24
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include "solver/preconditioner/uaamg_preconditioner.h"
#include "../test_utils.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>

// ── Test 1: PCG/UAAMG converges faster than CG ──
static void test_uaamg_faster_than_cg() {
    test_header("PCG/UAAMG faster than CG");

    Config cfg;
    cfg.scenario = "karman";
    cfg.NX = 64; cfg.NY = std::max(16, cfg.NX / 4);
    cfg.Lx = 4.0; cfg.Ly = 1.0; cfg.U_inf = 1.0; cfg.Re = 200;
    cfg.cyl_cx = 1.0; cfg.cyl_cy = 0.5; cfg.cyl_R = 0.1;
    cfg.t_end = 0.1;
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
    cfg.frame_skip = 100;
    cfg.solve_tol = 1e-6;
    int nsteps = (int)(cfg.t_end / cfg.dt);

    // CG with 200 iterations
    Config cfg_cg = cfg; cfg_cg.solve_iters = 200;
    auto cg = Factory::create("cg");
    ChorinSimulator sim_cg(cfg_cg, std::move(cg));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < nsteps; s++) sim_cg.step();
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_cg = std::chrono::duration<double>(t1 - t0).count();
    double max_div_cg = 0;
    for (int i = 1; i <= sim_cg.grid().nx; i++)
        for (int j = 1; j <= sim_cg.grid().ny; j++)
            if (!sim_cg.grid().is_solid(i,j))
                max_div_cg = std::max(max_div_cg, std::abs(sim_cg.grid().divergence(i,j)));

    // UAAMG with 50 iterations
    Config cfg_ua = cfg; cfg_ua.solve_iters = 50;
    auto ua = Factory::create("pcg_uaamg");
    ChorinSimulator sim_ua(cfg_ua, std::move(ua));
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < nsteps; s++) sim_ua.step();
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_ua = std::chrono::duration<double>(t3 - t2).count();
    double max_div_ua = 0;
    for (int i = 1; i <= sim_ua.grid().nx; i++)
        for (int j = 1; j <= sim_ua.grid().ny; j++)
            if (!sim_ua.grid().is_solid(i,j))
                max_div_ua = std::max(max_div_ua, std::abs(sim_ua.grid().divergence(i,j)));

    std::cout << "  CG(200):   time=" << std::fixed << std::setprecision(4) << t_cg
              << "s  max_div=" << std::scientific << max_div_cg << "\n";
    std::cout << "  UAAMG(50): time=" << t_ua
              << "s  max_div=" << max_div_ua << "\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(1) << t_cg/t_ua << "x\n";

    check(max_div_ua < 5e-3, "UAAMG passes divergence check");
    check(t_ua < t_cg, "UAAMG is faster than CG");

    test_summary();
}

// ── Test 2: PCG/UAAMG passes Karman divergence check ──
static void test_karman_uaamg() {
    test_header("PCG/UAAMG on Karman vortex street");

    Config cfg;
    cfg.scenario = "karman";
    cfg.NX = 128; cfg.NY = std::max(16, cfg.NX / 4);
    cfg.Lx = 4.0; cfg.Ly = 1.0; cfg.U_inf = 1.0; cfg.Re = 200;
    cfg.cyl_cx = 1.0; cfg.cyl_cy = 0.5; cfg.cyl_R = 0.1;
    cfg.t_end = 0.5;
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
    cfg.solve_iters = 50;
    cfg.solve_tol = 1e-6;
    cfg.frame_skip = 100;
    cfg.out_dir = "/tmp/karman_test";

    auto solver = Factory::create("pcg_uaamg");
    ChorinSimulator sim(cfg, std::move(solver));
    int nsteps = (int)(cfg.t_end / cfg.dt);

    for (int s = 0; s < nsteps; s++) sim.step();

    const Grid& g = sim.grid();
    double max_div = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i,j))
                max_div = std::max(max_div, std::abs(g.divergence(i,j)));

    std::cout << "  Steps=" << nsteps << "  max_div=" << std::scientific << max_div << "\n";
    check(max_div < 5e-3, "PCG/UAAMG divergence bounded");

    test_summary();
}

// ── Test 3: Compare PCG/UAAMG convergence with CG ──
static void test_compare_cg() {
    test_header("PCG/UAAMG vs CG convergence");

    Config cfg;
    cfg.scenario = "karman";
    cfg.NX = 64; cfg.NY = std::max(16, cfg.NX / 4);
    cfg.Lx = 4.0; cfg.Ly = 1.0; cfg.U_inf = 1.0; cfg.Re = 200;
    cfg.cyl_cx = 1.0; cfg.cyl_cy = 0.5; cfg.cyl_R = 0.1;
    cfg.t_end = 0.1;
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
    cfg.frame_skip = 100;

    int nsteps = (int)(cfg.t_end / cfg.dt);

    // CG reference
    Config cfg_cg = cfg;
    cfg_cg.solve_iters = 500;
    cfg_cg.solve_tol = 1e-10;
    auto cg_solver = Factory::create("cg");
    ChorinSimulator sim_cg(cfg_cg, std::move(cg_solver));
    for (int s = 0; s < nsteps; s++) sim_cg.step();
    const Grid& g_cg = sim_cg.grid();
    double max_div_cg = 0;
    for (int i = 1; i <= g_cg.nx; i++)
        for (int j = 1; j <= g_cg.ny; j++)
            if (!g_cg.is_solid(i,j))
                max_div_cg = std::max(max_div_cg, std::abs(g_cg.divergence(i,j)));

    // UAAMG with 50 iterations
    Config cfg_ua = cfg;
    cfg_ua.solve_iters = 50;
    cfg_ua.solve_tol = 1e-10;
    auto ua_solver = Factory::create("pcg_uaamg");
    ChorinSimulator sim_ua(cfg_ua, std::move(ua_solver));
    for (int s = 0; s < nsteps; s++) sim_ua.step();
    const Grid& g_ua = sim_ua.grid();
    double max_div_ua = 0;
    for (int i = 1; i <= g_ua.nx; i++)
        for (int j = 1; j <= g_ua.ny; j++)
            if (!g_ua.is_solid(i,j))
                max_div_ua = std::max(max_div_ua, std::abs(g_ua.divergence(i,j)));

    std::cout << "  CG(500):    max_div=" << std::scientific << max_div_cg << "\n";
    std::cout << "  UAAMG(50):  max_div=" << std::scientific << max_div_ua << "\n";

    check(max_div_ua < 5e-3, "UAAMG(50) passes divergence threshold");
    check(max_div_ua < 1e-3, "UAAMG(50) good accuracy");

    test_summary();
}

int main() {
    test_uaamg_faster_than_cg();
    test_karman_uaamg();
    test_compare_cg();
    return test_summary();
}

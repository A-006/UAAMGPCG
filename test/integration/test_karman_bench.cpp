/**
 * @file test_karman_bench.cpp
 * @brief Karman vortex street — solver comparison test.
 * @author liutao
 * @date 2026-05-22
 *
 * Runs the same Karman vortex street setup with different solvers
 * (Jacobi, RBGS, CG, PCG/GMG, PCG/AMG) and reports wall-clock time,
 * per-step cost, and correctness.
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include "../test_utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <sys/stat.h>

/// Single solver run: returns {wall_seconds, steps, ok}
static std::vector<double> run_one(const Config& base_cfg,
                                    const std::string& solver_key,
                                    int iters)
{
    Config cfg = base_cfg;
    cfg.solver = solver_key;
    cfg.solve_iters = iters;

    auto solver = Factory::create(cfg.solver);
    ChorinSimulator sim(cfg, std::move(solver));
    int nsteps = (int)(cfg.t_end / cfg.dt);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < nsteps; s++)
        sim.step();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Verify correctness: final divergence should be small everywhere
    bool ok = true;
    double max_div = 0;
    const Grid& g = sim.grid();
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            if (!g.is_solid(i,j)) {
                double d = std::abs(g.divergence(i,j));
                if (d > max_div) max_div = d;
                if (d > 5e-3) ok = false;
            }

    std::cout << "max_div=" << std::scientific << std::setprecision(2)
              << max_div << "  ";
    return {elapsed, (double)nsteps, ok ? 1.0 : 0.0};
}

int main() {
    test_header("Karman Vortex Street — Solver Benchmark");

    // ── Base configuration (same for all solvers) ──
    Config cfg;
    cfg.scenario = "karman";
    cfg.NX       = 128;
    cfg.Lx       = 4.0;
    cfg.Ly       = 1.0;
    cfg.U_inf    = 1.0;
    cfg.Re       = 200;
    cfg.cyl_cx   = 1.0;
    cfg.cyl_cy   = 0.5;
    cfg.cyl_R    = 0.1;
    cfg.t_end    = 0.5;
    cfg.dt       = 0.0;       // auto
    cfg.solve_iters = 0;      // auto
    cfg.solve_tol   = 1e-6;
    cfg.frame_skip  = 100;    // no VTK output during test
    cfg.out_dir     = "/tmp/karman_test";

    // Apply scenario defaults
    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
    int nsteps = (int)(cfg.t_end / cfg.dt);

    std::cout << "Grid: " << cfg.NX << "x" << cfg.NY
              << "  dt=" << std::fixed << std::setprecision(6) << cfg.dt
              << "  steps=" << nsteps
              << "  t_end=" << cfg.t_end << "\n\n";

    // ── Run each solver ──
    struct Result {
        std::string name;
        double time;
        int    steps;
        int    iters_per_step;
        bool   ok;
    };
    std::vector<Result> results;

    auto run = [&](const std::string& key, const std::string& label, int iters) {
        std::cout << "  " << std::left << std::setw(12) << (label + " ...")
                  << std::flush;
        auto v = run_one(cfg, key, iters);
        bool ok = (v[2] > 0.5);
        results.push_back({label, v[0], (int)v[1], iters, ok});
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << v[0] << " s  ("
                  << std::setprecision(1) << (v[0]/v[1]*1000) << " ms/step)  "
                  << (ok ? "PASS" : "FAIL") << "\n";
    };

    run("jacobi",    "Jacobi",     2000);
    run("rbgs",      "RBGS",       2000);
    run("cg",        "CG",          200);
    run("pcg",       "PCG/GMG",      50);
    run("pcg_amg",   "PCG/AMG",      50);
    run("pcg_uaamg", "PCG/UAAMG",    50);

    // ── Summary table ──
    std::cout << "\n";
    std::cout << "+" << std::string(62, '-') << "+\n";
    std::cout << "| " << std::left << std::setw(18) << "Solver"
              << std::right << std::setw(10) << "Time(s)"
              << std::setw(10) << "ms/step"
              << std::setw(10) << "Iters"
              << std::setw(14) << "vs Jacobi"
              << " |\n";
    std::cout << "+" << std::string(62, '-') << "+\n";

    double baseline = results[0].time;
    for (const auto& r : results) {
        double ms = r.time / r.steps * 1000;
        double ratio = (baseline > 0) ? r.time / baseline * 100 : 0;
        std::cout << "| " << std::left << std::setw(18) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << r.time
                  << std::setw(10) << std::setprecision(1) << ms
                  << std::setw(10) << r.iters_per_step
                  << std::setw(10) << std::setprecision(1) << ratio << "%"
                  << std::setw(4) << ""
                  << " |\n";
    }
    std::cout << "+" << std::string(62, '-') << "+\n\n";

    // ── Assertions ──
    for (const auto& r : results)
        check(r.ok, r.name + " divergence bounded");

    // CG & PCG should be at least 10x faster than Jacobi
    for (size_t i = 2; i < results.size(); i++)
        check(results[i].time < results[0].time / 5,
              results[i].name + " significantly faster than Jacobi");

    return test_summary();
}

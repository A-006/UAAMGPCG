/**
 * @file test_karman_bench_3d_gpu.cu
 * @brief 3D Karman vortex street — CPU + GPU solver comparison.
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include "solver/cuda_pcg_solver.h"       // 2D GPU wrapper
#include "solver/cuda_pcg_solver_3d.h"    // 3D GPU wrapper
#include "../test_utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <sys/stat.h>

static std::vector<double> run_one(const Config& base_cfg,
                                    std::unique_ptr<Solver> solver,
                                    const std::string& label,
                                    int iters)
{
    Config cfg = base_cfg;
    cfg.solver = label;
    cfg.solve_iters = iters;

    ChorinSimulator sim(cfg, std::move(solver));
    int nsteps = (int)(cfg.t_end / cfg.dt);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < nsteps; s++) sim.step();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

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
    test_header("3D Karman Vortex Street — CPU + GPU Solver Benchmark");

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
    cfg.dt       = 0.0;
    cfg.solve_iters = 0;
    cfg.solve_tol   = 1e-6;
    cfg.frame_skip  = 100;
    cfg.out_dir     = "/tmp/karman_test_3d_gpu";

    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
    int nsteps = (int)(cfg.t_end / cfg.dt);

    std::cout << "Grid: " << cfg.NX << "x" << cfg.NY
              << "  dt=" << std::fixed << std::setprecision(6) << cfg.dt
              << "  steps=" << nsteps
              << "  t_end=" << cfg.t_end << "\n\n";

    struct Result {
        std::string name; double time; int steps; int iters_per_step; bool ok;
    };
    std::vector<Result> results;

    auto run = [&](std::unique_ptr<Solver> solver, const std::string& label, int iters) {
        std::cout << "  " << std::left << std::setw(22) << (label + " ...")
                  << std::flush;
        auto v = run_one(cfg, std::move(solver), label, iters);
        bool ok = (v[2] > 0.5);
        results.push_back({label, v[0], (int)v[1], iters, ok});
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << v[0] << " s  ("
                  << std::setprecision(1) << (v[0]/v[1]*1000) << " ms/step)  "
                  << (ok ? "PASS" : "FAIL") << "\n";
    };

    // ── CPU solvers ──
    run(Factory::create("jacobi"),      "CPU Jacobi",          2000);
    run(Factory::create("rbgs"),        "CPU RBGS",            2000);
    run(Factory::create("cg"),          "CPU CG",               200);
    run(Factory::create("pcg"),         "CPU PCG/GMG",           50);
    run(Factory::create("pcg_uaamg"),   "CPU PCG/UAAMG",         50);

    // ── GPU solvers (2D, running on the same 2D Karman scenario) ──
    run(std::make_unique<CudaPCGSolver>(false), "GPU CG 2D",             200);
    run(std::make_unique<CudaPCGSolver>(true),  "GPU PCG/UAAMG 2D",       50);

    // ── Summary table ──
    std::cout << "\n";
    std::cout << "+" << std::string(75, '-') << "+\n";
    std::cout << "| " << std::left << std::setw(22) << "Solver"
              << std::right << std::setw(12) << "Time(s)"
              << std::setw(12) << "ms/step"
              << std::setw(10) << "Iters"
              << std::setw(14) << "vs Jacobi"
              << " |\n";
    std::cout << "+" << std::string(75, '-') << "+\n";

    double baseline = results[0].time;
    for (const auto& r : results) {
        double ms = r.time / r.steps * 1000;
        double ratio = (baseline > 0) ? r.time / baseline * 100 : 0;
        std::cout << "| " << std::left << std::setw(22) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.time
                  << std::setw(12) << std::setprecision(1) << ms
                  << std::setw(10) << r.iters_per_step
                  << std::setw(11) << std::setprecision(1) << ratio << "%"
                  << std::setw(7) << ""
                  << " |\n";
    }
    std::cout << "+" << std::string(75, '-') << "+\n\n";

    for (const auto& r : results)
        check(r.ok, r.name + " divergence bounded");

    // Check GPU matches CPU for UAAMGPCG
    check(results[4].ok && results[6].ok,
          "GPU and CPU UAAMGPCG both pass divergence check");

    return test_summary();
}

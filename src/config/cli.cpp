#include "config/cli.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace config {
namespace {

// Stationary relaxation (jacobi/rbgs) needs many cheap sweeps to converge;
// Krylov / multigrid solvers need only a handful of expensive iterations.
int default_solve_iters(std::string_view solver) {
    return (solver == "jacobi" || solver == "rbgs") ? 2000 : 50;
}

// Fill in the domain extent, grid aspect, time step and output directory that
// define a named scenario. Returns false for an unknown scenario.
bool apply_scenario(Config& cfg) {
    if (cfg.scenario == "karman") {
        cfg.Lx      = 4.0;
        cfg.Ly      = 1.0;
        cfg.NY      = std::max(cfg.NX / 4, 16);
        cfg.dt      = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf; // CFL ≈ 0.5
        cfg.out_dir = "output_karman";
    } else if (cfg.scenario == "smoke") {
        cfg.Lx      = 1.0;
        cfg.Ly      = 1.0;
        cfg.NY      = cfg.NX;
        cfg.dt      = 0.005;
        cfg.out_dir = "output_smoke";
    } else {
        return false;
    }
    cfg.solve_iters = default_solve_iters(cfg.solver);
    return true;
}

} // namespace

std::optional<Config> parse_cli(int argc, char* argv[]) {
    Config cfg;
    cfg.solver = "jacobi"; // CLI default (overridable by argv[4])

    if (argc > 1)
        cfg.scenario = argv[1];
    if (argc > 2)
        cfg.NX = std::atoi(argv[2]);
    if (argc > 3)
        cfg.t_end = std::atof(argv[3]);
    if (argc > 4)
        cfg.solver = argv[4];

    if (!apply_scenario(cfg)) {
        std::cerr << "Usage: lfm_2d [karman|smoke] [NX] [t_end] "
                     "[jacobi|rbgs|cg|pcg|pcg_gmg|pcg_amg|pcg_uaamg]\n";
        return std::nullopt;
    }
    return cfg;
}

} // namespace config

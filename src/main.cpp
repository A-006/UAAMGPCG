#include "config/config.h"
#include "simulator/simulator.h"
#include "solver/solver_factory.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    Config cfg;
    std::string solver_arg = "jacobi";

    if (argc > 1) cfg.scenario = argv[1];
    if (argc > 2) cfg.NX = std::atoi(argv[2]);
    if (argc > 3) cfg.t_end = std::atof(argv[3]);
    if (argc > 4) solver_arg = argv[4];

    if (cfg.scenario == "karman") {
        cfg.Lx = 4.0; cfg.Ly = 1.0;
        cfg.NY = cfg.NX / 4;
        if (cfg.NY < 16) cfg.NY = 16;
        cfg.out_dir = "output_karman";
        cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
        cfg.solve_iters = (solver_arg == "jacobi" || solver_arg == "rbgs") ? 2000 : 50;
    } else if (cfg.scenario == "smoke") {
        cfg.Lx = 1.0; cfg.Ly = 1.0;
        cfg.NY = cfg.NX;
        cfg.out_dir = "output_smoke";
        cfg.dt = 0.005;
        cfg.solve_iters = (solver_arg == "jacobi" || solver_arg == "rbgs") ? 2000 : 50;
    } else {
        std::cerr << "Usage: lfm_2d [karman|smoke] [NX] [t_end] [jacobi|rbgs|cg|pcg]\n";
        return 1;
    }

    LFMSimulator sim(cfg, SolverFactory::create(solver_arg));
    sim.run();
    return 0;
}

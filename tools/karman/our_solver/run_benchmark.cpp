/**
 * Karman vortex street — our UAAMGPCG solver benchmark.
 * Outputs Cd/Cl time history for OpenFOAM comparison.
 *
 * Usage: cd tools/karman/our_solver && ./build_and_run.sh [NX] [t_end]
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include "force/force.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    int NX = 256;
    double TEND = 10.0;
    if (argc > 1) NX = std::atoi(argv[1]);
    if (argc > 2) TEND = std::atof(argv[2]);

    Config cfg;
    cfg.scenario = "karman"; cfg.NX = NX; cfg.Lx = 4.0; cfg.Ly = 1.0;
    cfg.U_inf = 1.0; cfg.Re = 200; cfg.cyl_cx = 1.0; cfg.cyl_cy = 0.5; cfg.cyl_R = 0.1;
    cfg.t_end = TEND; cfg.dt = 0.0; cfg.solve_iters = 50; cfg.solve_tol = 1e-6;
    cfg.frame_skip = 10000; cfg.out_dir = "output"; cfg.solver = "pcg_uaamg";
    cfg.NY = std::max(16, NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;

    int nsteps = (int)(cfg.t_end / cfg.dt);
    double D = 2.0 * cfg.cyl_R, U = cfg.U_inf;

    std::cout << "# Karman UAAMGPCG  NX=" << cfg.NX << " NY=" << cfg.NY
              << " dt=" << cfg.dt << " steps=" << nsteps
              << " t_end=" << cfg.t_end << "\n";
    std::cout << "# D=" << D << " Re=" << cfg.Re << " U_inf=" << U << "\n";
    std::cout << "# time,Cd,Cl,max_div\n";

    auto solver = Factory::create(cfg.solver);
    LFMSimulator sim(cfg, std::move(solver));

    std::ofstream fout("force_history.csv");
    fout << "time,Cd,Cl,max_div\n";
    fout << std::scientific << std::setprecision(10);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < nsteps; s++) {
        sim.step();
        double t = (s + 1) * cfg.dt;

        const Grid& g = sim.grid();
        double max_div = 0;
        for (int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++)
            if(!g.is_solid(i,j)) max_div = std::max(max_div, std::abs(g.divergence(i,j)));

        auto force = computeForce(g, cfg.dt, U, cfg.Re, cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
        double Cd = force.Cd(U, D), Cl = force.Cl(U, D);

        fout << t << "," << Cd << "," << Cl << "," << max_div << "\n";

        if (s % (nsteps/20) == 0) {
            std::cout << t << "," << Cd << "," << Cl << "," << max_div << "\n";
            // Also print to stdout for monitoring
            std::cerr << "  [" << (100*s/nsteps) << "%] t=" << t
                      << " Cd=" << Cd << " Cl=" << Cl << " max_div=" << max_div << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "Total: " << elapsed << " s (" << (elapsed/nsteps*1000) << " ms/step)\n";

    // Quick stats
    double Cd_sum=0, Cd_max=-1e30, Cd_min=1e30;
    int n_avg = 0;
    // Re-read file for stats
    fout.close();
    std::ifstream fin("force_history.csv");
    std::string line;
    std::getline(fin, line); // skip header
    double tt, cd, cl, md;
    std::vector<double> Cd_hist, Cl_hist, t_hist;
    while (fin >> tt) {
        fin.ignore(1); fin >> cd; fin.ignore(1); fin >> cl; fin.ignore(1); fin >> md;
        if (tt > TEND * 0.3) { // last 70%
            Cd_sum += cd; Cd_max=std::max(Cd_max,cd); Cd_min=std::min(Cd_min,cd);
            n_avg++;
            Cd_hist.push_back(cd); Cl_hist.push_back(cl); t_hist.push_back(tt);
        }
    }
    double Cd_mean = n_avg>0 ? Cd_sum/n_avg : 0;
    double St = estimateStrouhal(t_hist, Cl_hist, U, D);
    std::cerr << "Cd_mean=" << Cd_mean << " Cd_range=[" << Cd_min << "," << Cd_max << "] St=" << St << "\n";

    return 0;
}

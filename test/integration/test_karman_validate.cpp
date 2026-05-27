/**
 * @file test_karman_validate.cpp
 * @brief Karman vortex street validation — Cd/Cl/St output for OpenFOAM comparison.
 *
 * Re=200, cylinder D=0.2 centered at (1.0, 0.5) in domain [0,4]×[0,1].
 * Expected (literature): St ≈ 0.19-0.20, Cd_mean ≈ 1.3-1.4, Cl_rms ≈ 0.3-0.5
 *
 * Reference: Schäfer & Turek 1996, Rajani 2009
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator_base.h"
#include "simulator/simulator.h"
#include "lfm/lfm_simulator.h"
#include "solver/factory.h"
#include "force/force.h"
#include "../test_utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    int NX = 128;
    double TEND = 5.0;
    if (argc > 1) NX = std::atoi(argv[1]);
    if (argc > 2) TEND = std::atof(argv[2]);

    Config cfg;
    cfg.scenario = "karman";
    cfg.NX       = NX;
    cfg.Lx       = 4.0;
    cfg.Ly       = 1.0;
    cfg.U_inf    = 1.0;
    cfg.Re       = 200;
    cfg.cyl_cx   = 1.0;
    cfg.cyl_cy   = 0.5;
    cfg.cyl_R    = 0.1;
    cfg.t_end    = TEND;
    cfg.dt       = 0.0;    // auto
    cfg.solve_iters = (cfg.time_integrator=="lfm") ? 200 : 100;
    cfg.solve_tol   = 1e-8;
    cfg.frame_skip  = (cfg.time_integrator=="lfm") ? 1 : 1000;
    cfg.out_dir     = "/tmp/karman_validate";
    cfg.solver      = "pcg_uaamg";
    cfg.time_integrator = "chorin";  // "chorin" or "lfm"
    if (argc > 3) cfg.time_integrator = argv[3];

    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;

    // Compute steps: LFM advances lfm_cycle_steps*dt per step()
    double dt_per_step = (cfg.time_integrator == "lfm")
        ? cfg.dt * cfg.lfm_cycle_steps : cfg.dt;
    int nsteps = (int)(cfg.t_end / dt_per_step);

    double D = 2.0 * cfg.cyl_R;
    double U = cfg.U_inf;

    test_header("Karman Vortex Street Validation (Re=200)");
    std::cout << "Grid: " << cfg.NX << "x" << cfg.NY
              << "  dt=" << cfg.dt << "  dt/step=" << dt_per_step
              << "  steps=" << nsteps << "  t_end=" << cfg.t_end
              << "  integrator=" << cfg.time_integrator << "\n";
    std::cout << "Cylinder: cx=" << cfg.cyl_cx << " cy=" << cfg.cyl_cy
              << " D=" << D << "  Re=" << cfg.Re << "\n";
    std::cout << "Reference: St=0.19-0.20, Cd_mean=1.3-1.4\n\n";

    auto solver = Factory::create(cfg.solver);
    std::unique_ptr<Simulator> sim;
    if (cfg.time_integrator == "lfm")
        sim = std::make_unique<LFMSimulator>(cfg, std::move(solver));
    else
        sim = std::make_unique<ChorinSimulator>(cfg, std::move(solver));

    std::vector<double> time_hist, Cd_hist, Cl_hist;

    // Skip initial transient (t < 2.0) for Strouhal calculation
    double t_transient = 2.0;
    bool collecting = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < nsteps; s++) {
        sim->step();
        double t = sim->time();

        if (t >= t_transient && !collecting) {
            collecting = true;
            std::cout << "  Transient done at t=" << t << ", collecting force data...\n";
        }

        if (collecting && s % 2 == 0) {  // sample every 2 steps
            const Grid& g = sim->grid();
            auto force = computeForce(g, cfg.dt, U, cfg.Re, cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
            time_hist.push_back(t);
            Cd_hist.push_back(force.Cd(U, D));
            Cl_hist.push_back(force.Cl(U, D));

            if (time_hist.size() % 100 == 0) {
                std::cout << "  t=" << std::fixed << std::setprecision(3) << t
                          << "  Cd=" << std::setprecision(4) << force.Cd(U, D)
                          << "  Cl=" << force.Cl(U, D) << "\n";
            }
        }

        // Progress
        if (nsteps <= 10 || s % (nsteps/10) == 0) {
            const Grid& g = sim->grid();
            double max_div = 0;
            for (int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++)
                if(!g.is_solid(i,j)) max_div = std::max(max_div, std::abs(g.divergence(i,j)));
            std::cout << "  [" << (100*s/nsteps) << "%] t=" << t
                      << " max_div=" << std::scientific << std::setprecision(2)
                      << max_div << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Total time: " << elapsed << " s ("
              << (elapsed/nsteps*1000) << " ms/step)\n";

    // ── Compute statistics ──
    if (Cd_hist.size() < 10) {
        std::cout << "  ERROR: not enough data points\n";
        return 1;
    }

    double Cd_sum = 0, Cl_sum = 0, Cl_sq_sum = 0;
    double Cd_max = -1e30, Cd_min = 1e30, Cl_max = -1e30, Cl_min = 1e30;
    for (size_t i = Cd_hist.size()/3; i < Cd_hist.size(); i++) {  // last 2/3
        Cd_sum += Cd_hist[i];
        Cl_sum += Cl_hist[i];
        Cl_sq_sum += Cl_hist[i] * Cl_hist[i];
        Cd_max = std::max(Cd_max, Cd_hist[i]);
        Cd_min = std::min(Cd_min, Cd_hist[i]);
        Cl_max = std::max(Cl_max, Cl_hist[i]);
        Cl_min = std::min(Cl_min, Cl_hist[i]);
    }
    size_t n_avg = Cd_hist.size() - Cd_hist.size()/3;
    double Cd_mean = Cd_sum / n_avg;
    double Cl_rms = std::sqrt(Cl_sq_sum / n_avg);

    // Strouhal number
    double St = estimateStrouhal(time_hist, Cl_hist, U, D);

    // ── Write force history to file ──
    std::string outfile = "/tmp/karman_validate/force_history.csv";
    writeForceHistory(outfile, time_hist, Cd_hist, Cl_hist);
    std::cout << "  Force history written to: " << outfile << "\n";

    // ── Summary ──
    std::cout << "\n";
    std::cout << "+" << std::string(55, '-') << "+\n";
    std::cout << "| " << std::left << std::setw(30) << "Quantity"
              << std::right << std::setw(12) << "Our Solver"
              << std::setw(12) << "Literature"
              << " |\n";
    std::cout << "+" << std::string(55, '-') << "+\n";
    std::cout << "| " << std::left << std::setw(30) << "Cd_mean"
              << std::right << std::fixed << std::setprecision(4) << std::setw(12) << Cd_mean
              << std::setw(12) << "1.30-1.40"
              << " |\n";
    std::cout << "| " << std::left << std::setw(30) << "Cd_amplitude"
              << std::right << std::setw(12) << (Cd_max - Cd_min)/2.0
              << std::setw(12) << "~0.02"
              << " |\n";
    std::cout << "| " << std::left << std::setw(30) << "Cl_rms"
              << std::right << std::setw(12) << Cl_rms
              << std::setw(12) << "0.30-0.50"
              << " |\n";
    std::cout << "| " << std::left << std::setw(30) << "Cl_amplitude"
              << std::right << std::setw(12) << (Cl_max - Cl_min)/2.0
              << std::setw(12) << "~0.5-1.0"
              << " |\n";
    std::cout << "| " << std::left << std::setw(30) << "Strouhal number"
              << std::right << std::setw(12) << St
              << std::setw(12) << "0.19-0.20"
              << " |\n";
    std::cout << "+" << std::string(55, '-') << "+\n\n";

    // ── Pass/Fail checks ──
    check(std::abs(Cd_mean - 1.35) < 0.5, "Cd_mean ~1.35 (within 0.5)");
    check(std::abs(St - 0.195) < 0.15, "St ~0.195 (within 0.15, grid-limited)");
    check(Cl_rms < 2.0, "Cl finite");

    return test_summary();
}

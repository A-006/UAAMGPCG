/**
 * @file test_karman_validate_gpu.cu
 * @brief Karman validation with GPU solvers (UAAMG-preconditioned PCG on CUDA).
 *
 * Same as test_karman_validate.cpp but links cuda_uaamg_lib for GPU acceleration.
 * Usage: build/test_karman_validate_gpu [NX] [TEND] [chorin|lfm] [cpu|gpu]
 */
#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator_base.h"
#include "simulator/simulator.h"
#include "lfm/lfm_simulator.h"
#include "solver/factory.h"
#include "solver/cuda_pcg_solver.h"
#include "force/force.h"
#include "io/vtk_writer.h"
#include "../test_utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>

int main(int argc, char** argv) {
    int NX = 128;
    double TEND = 5.0;
    bool use_gpu = true;
    if (argc > 1) NX = std::atoi(argv[1]);
    if (argc > 2) TEND = std::atof(argv[2]);

    Config cfg;
    cfg.scenario = "karman";
    cfg.NX       = NX;
    cfg.Lx       = 8.0; cfg.Ly = 2.0; cfg.U_inf = 1.0;        // larger domain (blockage 10%)
    cfg.Re = 200; cfg.cyl_cx = 2.0; cfg.cyl_cy = 1.0; cfg.cyl_R = 0.1;
    cfg.t_end = TEND; cfg.dt = 0.0;
    cfg.solve_iters = 100; cfg.solve_tol = 1e-8;
    cfg.frame_skip = (cfg.time_integrator == "lfm") ? 1 : 10; cfg.out_dir = "/tmp/karman_lfm_vtk";
    cfg.solver = "pcg_uaamg";
    cfg.time_integrator = "chorin";
    cfg.lfm_cycle_steps = 10;
    cfg.solve_iters = 200;
    cfg.cylinder_type = "stair";
    if (argc > 3) cfg.time_integrator = argv[3];
    if (argc > 4) cfg.cylinder_type = argv[4];
    if (argc > 5) use_gpu = (std::string(argv[5]) == "gpu");

    cfg.NY = std::max(16, cfg.NX / 4);
    cfg.dt = (cfg.time_integrator == "lfm" ? 0.25 : 0.5) * (cfg.Lx / cfg.NX) / cfg.U_inf;
    double dt_per_step = (cfg.time_integrator == "lfm") ? cfg.dt * cfg.lfm_cycle_steps : cfg.dt;
    int nsteps = (int)(cfg.t_end / dt_per_step);

    double D = 2.0 * cfg.cyl_R;
    double U = cfg.U_inf;

    test_header("Karman Vortex Street Validation (Re=200)");
    std::cout << "Grid: " << cfg.NX << "x" << cfg.NY
              << "  dt=" << cfg.dt << "  dt/step=" << dt_per_step
              << "  steps=" << nsteps << "  t_end=" << cfg.t_end
              << "  integrator=" << cfg.time_integrator
              << "  solver=" << (use_gpu ? "GPU(UAAMG)" : "CPU(UAAMG)") << "\n";
    std::cout << "Cylinder: cx=" << cfg.cyl_cx << " cy=" << cfg.cyl_cy
              << " D=" << D << "  Re=" << cfg.Re << "\n";
    std::cout << "Reference: St=0.19-0.20, Cd_mean=1.3-1.4\n\n";

    // Create solver
    std::unique_ptr<Solver> solver;
    if (use_gpu)
        solver = std::make_unique<CudaPCGSolver>(true);
    else
        solver = Factory::create(cfg.solver);

    // Create simulator
    std::unique_ptr<Simulator> sim;
    if (cfg.time_integrator == "lfm")
        sim = std::make_unique<LFMSimulator>(cfg, std::move(solver));
    else
        sim = std::make_unique<ChorinSimulator>(cfg, std::move(solver));

    // ── Small initial v-perturbation to break y-symmetry (kick into shedding mode) ──
    {
        Grid& g = const_cast<Grid&>(sim->grid());
        double eps = 0.01 * cfg.U_inf;
        for (int j = 1; j <= cfg.NY; j++) {
            for (int i = 1; i <= cfg.NX; i++) {
                double x = (i - 0.5) * g.dx;
                double y = (j - 0.5) * g.dy;
                if (x > cfg.cyl_cx + cfg.cyl_R && x < cfg.cyl_cx + 5.0 * cfg.cyl_R)
                    g.v_at(i, j-1) += eps * std::sin(M_PI * (y - cfg.cyl_cy) / cfg.cyl_R);
            }
        }
    }

    std::vector<double> time_hist, Cd_hist, Cl_hist;
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

        if (collecting && s % 2 == 0) {
            const Grid& g = sim->grid();
            auto force = computeForce(g, cfg.dt, U, cfg.Re, cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
            time_hist.push_back(t);
            Cd_hist.push_back(force.Cd(U, D));
            Cl_hist.push_back(force.Cl(U, D));
        }

        if (nsteps <= 10 || s % (nsteps/10) == 0) {
            const Grid& g = sim->grid();
            double max_div = 0;
            for (int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++)
                if(!g.is_solid(i,j)) max_div = std::max(max_div, std::abs(g.divergence(i,j)));
            std::cout << "  [" << (100*s/nsteps) << "%] t=" << t
                      << " max_div=" << std::scientific << std::setprecision(2) << max_div << "\n";
        }
        // Write VTK every 4 steps for LFM visualization (avoid temporal aliasing)
        if (cfg.time_integrator == "lfm" && s % 4 == 0) {
            VtkWriter::write(const_cast<Grid&>(sim->grid()), s/4, cfg);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Total time: " << std::scientific << std::setprecision(4) << elapsed << " s ("
              << std::setprecision(1) << (elapsed/nsteps*1000) << " ms/step)\n";

    // Force statistics
    if (Cd_hist.size() < 10) {
        std::cout << "  ERROR: not enough data points\n";
        return 1;
    }

    double Cd_sum=0, Cl_sq_sum=0, Cd_max=-1e30, Cd_min=1e30, Cl_max=-1e30, Cl_min=1e30;
    size_t start_i = Cd_hist.size()/3;
    for (size_t i=start_i; i<Cd_hist.size(); i++) {
        Cd_sum+=Cd_hist[i]; Cl_sq_sum+=Cl_hist[i]*Cl_hist[i];
        Cd_max=std::max(Cd_max,Cd_hist[i]); Cd_min=std::min(Cd_min,Cd_hist[i]);
        Cl_max=std::max(Cl_max,Cl_hist[i]); Cl_min=std::min(Cl_min,Cl_hist[i]);
    }
    double Cd_mean = Cd_sum/(Cd_hist.size()-start_i);
    double Cl_rms = std::sqrt(Cl_sq_sum/(Cd_hist.size()-start_i));
    double St = estimateStrouhal(time_hist, Cl_hist, U, D);

    std::cout << "\n+-------------------------------------------------------+\n";
    std::cout << "| Quantity                        Our Solver  Literature |\n";
    std::cout << "+-------------------------------------------------------+\n";
    std::cout << "| Cd_mean                         " << std::fixed << std::setprecision(4) << std::setw(8) << Cd_mean << "   1.30-1.40 |\n";
    std::cout << "| Cd_amplitude                    " << std::setw(8) << (Cd_max-Cd_min)/2.0 << "       ~0.02 |\n";
    std::cout << "| Cl_rms                          " << std::setw(8) << Cl_rms << "   0.30-0.50 |\n";
    std::cout << "| Cl_amplitude                    " << std::setw(8) << (Cl_max-Cl_min)/2.0 << "    ~0.5-1.0 |\n";
    std::cout << "| Strouhal number                 " << std::setw(8) << St << "   0.19-0.20 |\n";
    std::cout << "+-------------------------------------------------------+\n\n";

    check(std::abs(Cd_mean - 1.35) < 0.5, "Cd_mean ~1.35 (within 0.5)");
    check(std::abs(St - 0.195) < 0.15, "St ~0.195 (within 0.15, grid-limited)");
    check(Cl_rms < 2.0, "Cl finite");

    // Write force history
    {
        std::string outfile = "/tmp/karman_validate/force_history.csv";
        std::ofstream f(outfile);
        f << "# time,Cd,Cl\n";
        for (size_t i=0;i<time_hist.size();i++)
            f << time_hist[i] << "," << Cd_hist[i] << "," << Cl_hist[i] << "\n";
        std::cout << "  Force history written to: " << outfile << "\n";
    }

    return test_summary();
}

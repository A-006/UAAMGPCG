#include "simulator/simulator.h"
#include "advection/advection.h"
#include "boundary/boundary.h"
#include "pressure/pressure.h"
#include "io/vtk_writer.h"
#include "solver/factory.h"
#include <iostream>
#include <sys/stat.h>
#include <chrono>
#include <iomanip>

LFMSimulator::LFMSimulator(const Config& cfg, std::unique_ptr<Solver> solver)
    : cfg_(cfg), grid_(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly),
      prev_(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly), solver_(std::move(solver))
{
    // Initial conditions
    if (cfg_.scenario == "karman") {
        if (cfg_.cyl_R > 0)
            BoundaryConditions::setupCylinder(grid_, cfg_.cyl_cx, cfg_.cyl_cy, cfg_.cyl_R);
        for (int i = 0; i <= cfg_.NX; i++)
            for (int j = 1; j <= cfg_.NY; j++)
                grid_.u_at(i,j) = cfg_.U_inf;
    }
    apply_bc();
    prev_ = grid_;
}

void LFMSimulator::apply_bc() {
    if (cfg_.scenario == "karman") BoundaryConditions::applyKarman(grid_, cfg_.U_inf);
    else                           BoundaryConditions::applySmoke(grid_);
    BoundaryConditions::applySolid(grid_);
}

void LFMSimulator::apply_forces() {
    if (cfg_.scenario != "smoke") return;
    double buoyancy = 5.0;
    for (int i = 1; i <= grid_.nx; i++)
        for (int j = 1; j <= grid_.ny; j++)
            if (!grid_.is_solid(i,j))
                grid_.v_at(i,j) += cfg_.dt * buoyancy;
}

void LFMSimulator::advect() {
    Grid g_adv = prev_;
    for (int i = 1; i < grid_.nx; i++)
        for (int j = 1; j <= grid_.ny; j++) g_adv.u_at(i,j) = 0.0;
    for (int i = 1; i <= grid_.nx; i++)
        for (int j = 1; j < grid_.ny; j++) g_adv.v_at(i,j) = 0.0;

    AdvectionScheme::advect(prev_, g_adv, cfg_.dt);

    for (int i = 1; i < grid_.nx; i++)
        for (int j = 1; j <= grid_.ny; j++) grid_.u_at(i,j) = g_adv.u_at(i,j);
    for (int i = 1; i <= grid_.nx; i++)
        for (int j = 1; j < grid_.ny; j++) grid_.v_at(i,j) = g_adv.v_at(i,j);
}

void LFMSimulator::project() {
    PressureProjection::project(grid_, cfg_.dt, *solver_, cfg_.solve_iters, cfg_.solve_tol);
}

void LFMSimulator::step() {
    apply_forces();
    advect();
    apply_bc();
    project();
    apply_bc();
    prev_ = grid_;
    t_ += cfg_.dt;
    step_++;
}

void LFMSimulator::run() {
    mkdir(cfg_.out_dir.c_str(), 0755);

    std::cout << "+" << std::string(52, '-') << "+\n";
    std::cout << "| LFM 2D Fluid Simulation — " << cfg_.scenario
              << std::string(26 - (int)cfg_.scenario.size(), ' ') << "|\n";
    std::cout << "| grid: " << cfg_.NX << "x" << cfg_.NY
              << "  cells=" << cfg_.NX * cfg_.NY
              << std::string(16, ' ') << "|\n";
    std::cout << "| dx=" << std::setprecision(4) << (cfg_.Lx/cfg_.NX)
              << "  dy=" << (cfg_.Ly/cfg_.NY)
              << "  dt=" << cfg_.dt
              << "  t_end=" << cfg_.t_end << "        |\n";
    std::cout << "| Solver=" << solver_->name()
              << "  iters=" << cfg_.solve_iters << "        |\n";
    std::cout << "| output: " << cfg_.out_dir << "/frame_*.vtk |\n";
    std::cout << "+" << std::string(52, '-') << "+\n\n";

    int nsteps = (int)(cfg_.t_end / cfg_.dt);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < nsteps; s++) {
        step();
        if (s % cfg_.frame_skip == 0) {
            VtkWriter::printStatus(s, t_, grid_);
            VtkWriter::write(grid_, s / cfg_.frame_skip, cfg_);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nDone: " << nsteps << " steps in " << elapsed << " s";
    std::cout << " (" << (elapsed/nsteps*1000) << " ms/step)\n";
    std::cout << "Open " << cfg_.out_dir << "/frame_*.vtk in ParaView\n";
}

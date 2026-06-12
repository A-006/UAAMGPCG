#include "simulator/runner_3d.h"
#include "io/vtk_slim_3d.h"
#include "io/vtk_writer_3d.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace sim3d {

void run(Simulator3D& sim, const Config& cfg) {
    std::string backend  = cfg.sget("backend", "cpu"); // resolved by make_simulator
    int gpu_dev          = cfg.iget("gpu", -1);
    int n_cycles         = cfg.iget("cycles", 100);
    std::string vtk_mode = cfg.sget("vtk_mode", "slim");
    bool full_vtk        = (vtk_mode == "full");
    bool dump_vel        = cfg.iget("dump_vel", 0) != 0; // cell-centered velocity .raw per frame
    double dx            = cfg.Lx / cfg.NX;

    std::cout << "===================================================\n";
    std::cout << "  3D LFM (" << backend << ") — scenario: " << cfg.scenario << "\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  Grid " << cfg.NX << "x" << cfg.NY << "x" << cfg.NZ << "  dx=" << dx << "\n";
    std::cout << "  dt=" << cfg.dt << "  n=" << cfg.lfm_cycle_steps << "/cycle  cycles=" << n_cycles
              << "  Re=" << cfg.Re << "  clamp=" << (cfg.lfm_bfecc_clamp ? "on" : "off") << "\n";
    std::cout << "  bc=" << cfg.lfm_bc << "  vtk=" << vtk_mode << "  out=" << cfg.out_dir << "/"
              << (backend == "gpu" ? (gpu_dev >= 0 ? ("  gpu=" + std::to_string(gpu_dev)) : "")
                                   : ("  solver=" + cfg.solver))
              << "\n";
    if (backend == "cpu")
        std::cout << "  [note] CPU backend is a correctness reference — much slower than GPU;"
                     " use small grids.\n";
    std::cout << "===================================================\n\n";

    mkdir(cfg.out_dir.c_str(), 0755);

    auto write_frame = [&](int frame) {
        if (full_vtk)
            VtkWriter3D::write(sim.grid(), frame, cfg);
        else
            io3d::write_vort_vtk(sim.grid(), frame, cfg.out_dir);
        if (dump_vel)
            io3d::write_vel_raw(sim.grid(), frame, cfg.out_dir);
    };

    int frame = 0;
    write_frame(frame++);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int c = 1; c <= n_cycles; c++) {
        sim.step();
        if (c % cfg.frame_skip == 0 || c == n_cycles)
            write_frame(frame++);
        if (c % 5 == 0 || c == n_cycles)
            VtkWriter3D::printStatus(c, sim.time(), sim.grid());
    }
    auto t1   = std::chrono::high_resolution_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n  Done: " << n_cycles << " cycles in " << std::fixed << std::setprecision(1)
              << el << " s  (" << el / n_cycles << " s/cycle)\n";
    std::cout << "  Output: " << cfg.out_dir << "/frame_*.vtk\n";
}

} // namespace sim3d

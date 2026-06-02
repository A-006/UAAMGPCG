#include "simulator/runner.h"
#include "io/vtk_writer.h"
#include <chrono>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace sim {

void run(Simulator& sim, const Config& cfg, const RunOptions& opts) {
    mkdir(cfg.out_dir.c_str(), 0755);

    if (opts.print_banner) {
        const std::string rule = "+" + std::string(52, '-') + "+\n";
        std::cout << rule;
        std::cout << "| LFM 2D Fluid — " << cfg.scenario << "  grid:" << cfg.NX << "x" << cfg.NY
                  << "  dt=" << cfg.dt << "  t_end=" << cfg.t_end << " |\n";
        std::cout << rule << "\n";
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    int s     = 0; // step index
    int frame = 0; // output frame index
    while (sim.time() < cfg.t_end - 1e-9) {
        sim.step();

        if (s % cfg.frame_skip == 0) {
            if (opts.print_banner)
                VtkWriter::printStatus(s, sim.time(), sim.grid());
            if (opts.write_vtk)
                VtkWriter::write(sim.grid(), frame, cfg);
            for (const auto& obs : opts.observers)
                obs(frame, sim, cfg);
            frame++;
        }
        s++;
    }

    if (opts.print_banner) {
        const auto t1        = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        const double per     = s > 0 ? elapsed / s * 1000 : 0.0;
        std::cout << "\nDone: " << s << " steps in " << elapsed << " s (" << per << " ms/step)\n";
    }
}

} // namespace sim

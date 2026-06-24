#include "integrator/runner.h"
#include "mesh/grid.h"
#include "io/vtk_writer.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {
// Cell-centered 2D velocity (float32, C-order i-slowest, nx*ny) as vx/vy_<frame>.raw,
// for the author-slab cross-check (compare against a mid-z slice of the 3D reference).
void dump_vel_raw_2d(const Grid& g, int frame, const std::string& dir) {
    long n = (long)g.nx * g.ny;
    std::vector<float> vx(n), vy(n);
    long c = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            vx[c] = (float)(0.5 * (g.u_at(i, j) + g.u_at(i - 1, j)));
            vy[c] = (float)(0.5 * (g.v_at(i, j) + g.v_at(i, j - 1)));
            c++;
        }
    char p[512];
    auto wr = [&](const char* nm, const std::vector<float>& b) {
        std::snprintf(p, sizeof(p), "%s/%s_%05d.raw", dir.c_str(), nm, frame);
        std::ofstream f(p, std::ios::binary);
        f.write((const char*)b.data(), b.size() * sizeof(float));
    };
    wr("vx", vx);
    wr("vy", vy);
}
} // namespace

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

    const bool dump_vel = cfg.iget("dump_vel", 0) != 0;
    // Raw-dump frame index = number of completed cycles (one 2D LFM step() = one
    // cycle), so it aligns with the author runner's per-cycle frames. Frame 0 = IC.
    if (dump_vel)
        dump_vel_raw_2d(sim.grid(), 0, cfg.out_dir);

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
        if (dump_vel)
            dump_vel_raw_2d(sim.grid(), s + 1, cfg.out_dir);
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

// Simulation::run for every 2D simulator — drives the shared loop above.
void Simulator::run(const Config& cfg) {
    sim::run(*this, cfg);
}

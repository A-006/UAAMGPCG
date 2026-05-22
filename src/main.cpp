#include "lfm/config.h"
#include "lfm/grid.h"
#include "lfm/advection.h"
#include "lfm/boundary.h"
#include "lfm/pressure.h"
#include "lfm/vtk_io.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>

// LFM time step:
//   1. External forces (buoyancy)
//   2. Leapfrog advection (RK2 semi-Lagrangian)
//   3. Pressure projection (only once — core LFM advantage)
static void lfm_time_step(Grid& g, double dt, const Config& cfg,
                          const Grid* g_bc = nullptr) {
    if (cfg.scenario == "smoke") {
        double buoyancy = 5.0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j))
                    g.v_at(i,j) += dt * buoyancy;
    }

    const Grid& g_old = (g_bc ? *g_bc : g);
    Grid g_adv = g_old;

    for (int i = 1; i < g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            g_adv.u_at(i,j) = 0.0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j < g.ny; j++)
            g_adv.v_at(i,j) = 0.0;

    advect_velocity(g_old, g_adv, dt);

    for (int i = 1; i < g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            g.u_at(i,j) = g_adv.u_at(i,j);
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j < g.ny; j++)
            g.v_at(i,j) = g_adv.v_at(i,j);

    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);

    pressure_projection(g, dt, cfg);

    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);
}

int main(int argc, char* argv[]) {
    Config cfg;

    if (argc > 1) cfg.scenario = argv[1];
    if (argc > 2) cfg.NX = std::atoi(argv[2]);
    if (argc > 3) cfg.t_end = std::atof(argv[3]);
    if (argc > 4) {
        std::string s = argv[4];
        if (s == "jacobi") cfg.solver = Solver::Jacobi;
        else if (s == "rbgs") cfg.solver = Solver::RBGS;
        else if (s == "cg")   cfg.solver = Solver::CG;
        else if (s == "pcg")  cfg.solver = Solver::PCG;
    }

    if (cfg.scenario == "karman") {
        cfg.Lx = 4.0; cfg.Ly = 1.0;
        cfg.NY = cfg.NX / 4;
        if (cfg.NY < 16) cfg.NY = 16;
        cfg.out_dir = "output_karman";
        cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;
        cfg.solve_iters = (cfg.solver <= Solver::RBGS) ? 2000 : 50;
    } else if (cfg.scenario == "smoke") {
        cfg.Lx = 1.0; cfg.Ly = 1.0;
        cfg.NY = cfg.NX;
        cfg.out_dir = "output_smoke";
        cfg.dt = 0.005;
        cfg.solve_iters = (cfg.solver <= Solver::RBGS) ? 2000 : 50;
    } else {
        std::cerr << "Usage: lfm_2d [karman|smoke] [NX] [t_end] [jacobi|rbgs|cg|pcg]\n";
        return 1;
    }

    mkdir(cfg.out_dir.c_str(), 0755);

    std::cout << "+" << std::string(52, '-') << "+\n";
    std::cout << "| LFM 2D Fluid Simulation — " << cfg.scenario
              << std::string(26 - (int)cfg.scenario.size(), ' ') << "|\n";
    std::cout << "| grid: " << cfg.NX << "x" << cfg.NY
              << "  cells=" << cfg.NX * cfg.NY
              << std::string(16, ' ') << "|\n";
    std::cout << "| dx=" << std::setprecision(4) << (cfg.Lx/cfg.NX)
              << "  dy=" << (cfg.Ly/cfg.NY)
              << "  dt=" << cfg.dt
              << "  t_end=" << cfg.t_end << "        |\n";
    std::cout << "| Solver=" << solver_name(cfg.solver) << "  iters=" << cfg.solve_iters
              << "  output: " << cfg.out_dir << "/frame_*.vtk |\n";
    std::cout << "+" << std::string(52, '-') << "+\n\n";

    Grid g(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly);

    if (cfg.scenario == "karman") {
        setup_cylinder(g, cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
        for (int i = 0; i <= cfg.NX; i++)
            for (int j = 1; j <= cfg.NY; j++)
                g.u_at(i,j) = cfg.U_inf;
    }

    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);

    int nsteps = (int)(cfg.t_end / cfg.dt);
    auto t0 = std::chrono::high_resolution_clock::now();
    double t = 0.0;
    Grid g_prev = g;

    for (int step = 0; step < nsteps; step++) {
        lfm_time_step(g, cfg.dt, cfg, &g_prev);
        g_prev = g;
        t += cfg.dt;

        if (step % cfg.frame_skip == 0) {
            print_status(step, t, g);
            write_vtk(g, step / cfg.frame_skip, cfg);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\nDone: " << nsteps << " steps in " << elapsed << " s";
    std::cout << " (" << (elapsed/nsteps*1000) << " ms/step)\n";
    std::cout << "Open " << cfg.out_dir << "/frame_*.vtk in ParaView\n";

    return 0;
}

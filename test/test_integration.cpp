#include "config/config.h"
#include "core/grid.h"
#include "advection/advection.h"
#include "simulator/simulator.h"
#include "solver/poisson_jacobi.h"
#include "solver/poisson_rbgs.h"
#include "solver/poisson_cg.h"
#include "solver/poisson_pcg.h"
#include "test_utils.h"
#include <cmath>
#include <vector>
#include <memory>

static std::unique_ptr<PoissonSolver> make_solver(const std::string& name) {
    if (name == "jacobi") return std::make_unique<JacobiSolver>();
    if (name == "rbgs")   return std::make_unique<RBGSSolver>();
    if (name == "cg")     return std::make_unique<CGSolver>();
    if (name == "pcg")    return std::make_unique<PCGSolver>();
    return std::make_unique<JacobiSolver>();
}

int main() {
    test_header("LFM Integration Tests");

    // Test 1: Uniform inflow with each solver
    for (auto name : {"jacobi", "rbgs", "cg", "pcg"}) {
        Config cfg;
        cfg.scenario = "karman"; cfg.NX = 32; cfg.NY = 16;
        cfg.Lx = 4.0; cfg.Ly = 1.0; cfg.cyl_R = 0;
        cfg.dt = 0.5 * (cfg.Lx/cfg.NX) / cfg.U_inf;
        cfg.solve_iters = (name[0]=='j'||name[0]=='r') ? 500 : 30;
        cfg.solve_tol = 1e-6;

        LFMSimulator sim(cfg, make_solver(name));
        bool ok = true;
        for (int s = 0; s < 3; s++) {
            sim.step();
            const Grid& g = sim.grid();
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++)
                    if (!g.is_solid(i,j) && std::abs(g.divergence(i,j)) > 1e-4) ok = false;
        }
        check(ok, std::string(name) + ": uniform flow");
    }

    // Test 2: Cylinder with CG/PCG
    for (auto name : {"cg", "pcg"}) {
        Config cfg;
        cfg.scenario = "karman"; cfg.NX = 64; cfg.NY = 32;
        cfg.Lx = 4.0; cfg.Ly = 1.0;
        cfg.dt = 0.5 * (cfg.Lx/cfg.NX) / cfg.U_inf;
        cfg.solve_iters = 30; cfg.solve_tol = 1e-6;

        LFMSimulator sim(cfg, make_solver(name));
        bool ok = true;
        for (int s = 0; s < 5; s++) {
            sim.step();
            const Grid& g = sim.grid();
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i,j)) continue;
                    double uc = 0.5*(g.u_at(i-1,j)+g.u_at(i,j));
                    double vc = 0.5*(g.v_at(i,j-1)+g.v_at(i,j));
                    if (!std::isfinite(uc) || !std::isfinite(vc)) ok = false;
                }
        }
        check(ok, std::string(name) + ": cylinder flow bounded");
    }

    // Test 3: Smoke buoyancy with each solver
    for (auto name : {"jacobi", "rbgs", "cg", "pcg"}) {
        Config cfg;
        cfg.scenario = "smoke"; cfg.NX = 32; cfg.NY = 32;
        cfg.Lx = 1.0; cfg.Ly = 1.0; cfg.dt = 0.005;
        cfg.solve_iters = (name[0]=='j'||name[0]=='r') ? 1000 : 20;
        cfg.solve_tol = 1e-6;

        LFMSimulator sim(cfg, make_solver(name));
        bool ok = true;
        for (int s = 0; s < 10; s++) {
            sim.step();
            const Grid& g = sim.grid();
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i,j)) continue;
                    double uc = 0.5*(g.u_at(i-1,j)+g.u_at(i,j));
                    double vc = 0.5*(g.v_at(i,j-1)+g.v_at(i,j));
                    if (!std::isfinite(uc) || !std::isfinite(vc)) ok = false;
                }
        }
        check(ok, std::string(name) + ": smoke stable");
    }

    return test_summary();
}

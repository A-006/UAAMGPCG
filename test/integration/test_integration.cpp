#include "io/config.h"
#include "mesh/grid_2d.h"
#include "integrator/ops/advection_2d.h"
#include "integrator/chorin/chorin_simulator_2d.h"
#include "solver/relaxation/jacobi_2d.h"
#include "solver/relaxation/rbgs_2d.h"
#include "solver/krylov/pcg_2d.h"
#include "solver/preconditioner/identity_preconditioner_2d.h"
#include "solver/preconditioner/gmg_preconditioner_2d.h"
#include "../test_utils.h"
#include "../test_config.h"
#include <cmath>
#include <vector>
#include <memory>

static std::unique_ptr<Solver> make_solver(const std::string& name) {
    if (name == "jacobi")
        return std::make_unique<Jacobi>();
    if (name == "rbgs")
        return std::make_unique<RBGS>();
    if (name == "cg")
        return std::make_unique<PCG>(std::make_unique<IdentityPreconditioner>());
    if (name == "pcg")
        return std::make_unique<PCG>(std::make_unique<GMGPreconditioner>());
    return std::make_unique<Jacobi>();
}

int main() {
    test_header("LFM Integration Tests");

    // Test 1: Uniform inflow with each solver
    for (auto name : {"jacobi", "rbgs", "cg", "pcg"}) {
        Config cfg      = make_karman_config(32);
        cfg.cyl_R       = 0; // no cylinder: just uniform inflow
        cfg.solve_iters = (name[0] == 'j' || name[0] == 'r') ? 500 : 30;

        ChorinSimulator sim(cfg, make_solver(name));
        bool ok = true;
        for (int s = 0; s < 3; s++) {
            sim.step();
            const Grid& g = sim.grid();
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++)
                    if (!g.is_solid(i, j) && std::abs(g.divergence(i, j)) > 1e-4)
                        ok = false;
        }
        check(ok, std::string(name) + ": uniform flow");
    }

    // Test 2: Cylinder with CG/PCG
    for (auto name : {"cg", "pcg"}) {
        Config cfg      = make_karman_config(64);
        cfg.NY          = 32; // taller grid than the default NX/4 aspect
        cfg.solve_iters = 30;

        ChorinSimulator sim(cfg, make_solver(name));
        bool ok = true;
        for (int s = 0; s < 5; s++) {
            sim.step();
            const Grid& g = sim.grid();
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i, j))
                        continue;
                    double uc = 0.5 * (g.u_at(i - 1, j) + g.u_at(i, j));
                    double vc = 0.5 * (g.v_at(i, j - 1) + g.v_at(i, j));
                    if (!std::isfinite(uc) || !std::isfinite(vc))
                        ok = false;
                }
        }
        check(ok, std::string(name) + ": cylinder flow bounded");
    }

    // Test 3: Smoke buoyancy with each solver
    for (auto name : {"jacobi", "rbgs", "cg", "pcg"}) {
        Config cfg      = make_smoke_config(32);
        cfg.solve_iters = (name[0] == 'j' || name[0] == 'r') ? 1000 : 20;

        ChorinSimulator sim(cfg, make_solver(name));
        bool ok = true;
        for (int s = 0; s < 10; s++) {
            sim.step();
            const Grid& g = sim.grid();
            for (int i = 1; i <= g.nx; i++)
                for (int j = 1; j <= g.ny; j++) {
                    if (g.is_solid(i, j))
                        continue;
                    double uc = 0.5 * (g.u_at(i - 1, j) + g.u_at(i, j));
                    double vc = 0.5 * (g.v_at(i, j - 1) + g.v_at(i, j));
                    if (!std::isfinite(uc) || !std::isfinite(vc))
                        ok = false;
                }
        }
        check(ok, std::string(name) + ": smoke stable");
    }

    return test_summary();
}

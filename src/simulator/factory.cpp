#include "simulator/factory.h"
#include "simulator/simulator.h"
#include "lfm/lfm_simulator.h"
#include "solver/factory.h"
#include <stdexcept>

namespace SimulatorFactory {

std::unique_ptr<Solver> make_pressure_solver(const Config& cfg) {
    if (cfg.dim == 2)
        return Factory::create(cfg.solver);
    if (cfg.dim == 3)
        throw std::runtime_error(
            "make_pressure_solver: dim=3 path returns Solver3D; "
            "use Factory3D directly until the 3D Simulator is implemented.");
    throw std::runtime_error("make_pressure_solver: cfg.dim must be 2 or 3.");
}

std::unique_ptr<Simulator> create(const Config& cfg, std::unique_ptr<Solver> solver) {
    if (cfg.dim == 2) {
        if (cfg.time_integrator == "lfm")
            return std::make_unique<LFMSimulator>(cfg, std::move(solver));
        return std::make_unique<ChorinSimulator>(cfg, std::move(solver));
    }
    if (cfg.dim == 3) {
        throw std::runtime_error(
            "SimulatorFactory::create: dim=3 simulators (ChorinSimulator3D / "
            "LFMSimulator3D) are not yet implemented. 3D solver kernels exist "
            "under src/solver/{...}_3d.cpp; full 3D simulator integration is "
            "tracked as a follow-up.");
    }
    throw std::runtime_error("SimulatorFactory::create: cfg.dim must be 2 or 3.");
}

}  // namespace SimulatorFactory

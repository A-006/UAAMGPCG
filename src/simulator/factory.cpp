#include "simulator/factory.h"
#include "simulator/chorin_simulator.h"
#include "simulator/lfm_simulator.h"
#include "simulator/lfm_simulator_3d.h"
#include "simulator/simulator_3d.h"
#include "solver/factory.h"
#include "solver/factory_3d.h"
#include <stdexcept>

namespace SimulatorFactory {

std::unique_ptr<Solver> make_pressure_solver(const Config& cfg) {
    if (cfg.dim == 2)
        return Factory::create(cfg.solver);
    if (cfg.dim == 3)
        throw std::runtime_error("make_pressure_solver: dim=3 path returns Solver3D; "
                                 "use Factory3D directly until the 3D Simulator is implemented.");
    throw std::runtime_error("make_pressure_solver: cfg.dim must be 2 or 3.");
}

std::unique_ptr<Simulator> create(const Config& cfg) {
    return create(cfg, make_pressure_solver(cfg));
}

std::unique_ptr<Simulator> create(const Config& cfg, std::unique_ptr<Solver> solver) {
    if (cfg.dim == 2) {
        if (cfg.time_integrator == "lfm")
            return std::make_unique<LFMSimulator>(cfg, std::move(solver));
        return std::make_unique<ChorinSimulator>(cfg, std::move(solver));
    }
    if (cfg.dim == 3) {
        throw std::runtime_error("SimulatorFactory::create: dim=3 simulators (ChorinSimulator3D / "
                                 "LFMSimulator3D) are not yet implemented. 3D solver kernels exist "
                                 "under src/solver/{...}_3d.cpp; full 3D simulator integration is "
                                 "tracked as a follow-up.");
    }
    throw std::runtime_error("SimulatorFactory::create: cfg.dim must be 2 or 3.");
}

std::unique_ptr<Solver3D> make_pressure_solver_3d(const Config& cfg) {
    return Factory3D::create(cfg.solver);
}

std::unique_ptr<Simulator3D> create3d(const Config& cfg) {
    return create3d(cfg, make_pressure_solver_3d(cfg));
}

std::unique_ptr<Simulator3D> create3d(const Config& cfg, std::unique_ptr<Solver3D> solver) {
    if (cfg.time_integrator == "lfm")
        return std::make_unique<LFMSimulator3D>(cfg, std::move(solver));
    return std::make_unique<ChorinSimulator3D>(cfg, std::move(solver));
}

} // namespace SimulatorFactory

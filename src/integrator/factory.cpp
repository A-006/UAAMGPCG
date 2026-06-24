#include "integrator/factory.h"
#include "integrator/chorin/chorin_simulator_2d.h"
#include "integrator/lfm/lfm_simulator_2d.h"
#include "solver/factory_2d.h"

// The 2D simulator factory. 3D provisioning (backend GPU/CPU selection +
// construction + IC setup) lives in make_simulator_3d.cpp, not here.
namespace SimulatorFactory {

std::unique_ptr<Solver> make_pressure_solver(const Config& cfg) {
    return Factory::create(cfg.solver);
}

std::unique_ptr<Simulator> create(const Config& cfg) {
    return create(cfg, make_pressure_solver(cfg));
}

std::unique_ptr<Simulator> create(const Config& cfg, std::unique_ptr<Solver> solver) {
    if (cfg.time_integrator == "lfm")
        return std::make_unique<LFMSimulator>(cfg, std::move(solver));
    return std::make_unique<ChorinSimulator>(cfg, std::move(solver));
}

} // namespace SimulatorFactory

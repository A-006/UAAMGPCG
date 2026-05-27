#pragma once
#include "config/config.h"
#include "simulator/simulator_base.h"
#include "solver/solver.h"
#include <memory>

// ──────────────────────────────────────────────────────────────────
// SimulatorFactory — single entry point for both 2D and 3D simulators.
//
//   auto solver = make_pressure_solver(cfg);
//   auto sim    = SimulatorFactory::create(cfg, std::move(solver));
//
// Dispatch is based on cfg.dim and cfg.time_integrator:
//   dim=2, integrator="chorin" → ChorinSimulator        (2D)
//   dim=2, integrator="lfm"    → LFMSimulator           (2D, paper Alg. 1)
//   dim=3, integrator="chorin" → ChorinSimulator3D      (3D, planned)
//   dim=3, integrator="lfm"    → LFMSimulator3D         (3D, planned)
//
// 3D simulators are not implemented yet; calling create() with dim=3
// currently throws std::runtime_error.
// ──────────────────────────────────────────────────────────────────
namespace SimulatorFactory {

// Returns an appropriate Simulator (2D or 3D) based on cfg.
std::unique_ptr<Simulator> create(const Config& cfg, std::unique_ptr<Solver> pressure_solver);

// Returns a pressure solver matching cfg.dim and cfg.solver.
// Wraps Factory::create / Factory3D::create.
std::unique_ptr<Solver> make_pressure_solver(const Config& cfg);

}  // namespace SimulatorFactory

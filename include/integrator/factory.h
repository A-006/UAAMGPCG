#pragma once
#include "io/config.h"
#include "integrator/simulator_2d.h"
#include "solver/solver.h"
#include <memory>

// ──────────────────────────────────────────────────────────────────
// SimulatorFactory — builds the 2D simulator from cfg.
//
//   auto solver = make_pressure_solver(cfg);
//   auto sim    = SimulatorFactory::create(cfg, std::move(solver));
//
// Dispatch is on cfg.time_integrator:
//   integrator="chorin" → ChorinSimulator
//   integrator="lfm"    → LFMSimulator (paper Alg. 1)
//
// 3D is NOT here: 3D provisioning (GPU/CPU backend selection + construction +
// IC setup) lives in make_simulator_3d.cpp (scene3d::make_simulator).
// ──────────────────────────────────────────────────────────────────
namespace SimulatorFactory {

// Build the 2D simulator for cfg, constructing the pressure solver internally.
std::unique_ptr<Simulator> create(const Config& cfg);

// Same, with a caller-supplied solver (e.g. a GPU solver injected by a
// benchmark tool). The overload above delegates here.
std::unique_ptr<Simulator> create(const Config& cfg, std::unique_ptr<Solver> pressure_solver);

// Build the 2D pressure solver matching cfg.solver.
std::unique_ptr<Solver> make_pressure_solver(const Config& cfg);

} // namespace SimulatorFactory

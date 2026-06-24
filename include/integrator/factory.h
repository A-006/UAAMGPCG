#pragma once
#include "core/config.h"
#include "integrator/simulator_3d.h"
#include "integrator/simulator_2d.h"
#include "solver/solver.h"
#include "solver/solver_3d.h"
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

// Returns an appropriate Simulator (2D or 3D) based on cfg, building the
// matching pressure solver internally. This is the one-call entry point for
// the common case.
std::unique_ptr<Simulator> create(const Config& cfg);

// Same, but with a caller-supplied solver (e.g. a GPU solver injected by a
// benchmark tool). The overload above delegates here.
std::unique_ptr<Simulator> create(const Config& cfg, std::unique_ptr<Solver> pressure_solver);

// Returns a pressure solver matching cfg.dim and cfg.solver.
// Wraps Factory::create / Factory3D::create.
std::unique_ptr<Solver> make_pressure_solver(const Config& cfg);

// ── 3D entry points ───────────────────────────────────────────────
// Simulator3D is a separate base class from Simulator (the codebase keeps
// 2D/3D types parallel rather than templating), so the 3D simulators cannot
// be returned through the Simulator-typed create() above — these typed
// entry points mirror the 2D factory for the dim==3 case.
//
//   dim=3, integrator="chorin" → ChorinSimulator3D
//   dim=3, integrator="lfm"    → LFMSimulator3D
std::unique_ptr<Solver3D> make_pressure_solver_3d(const Config& cfg);
std::unique_ptr<Simulator3D> create3d(const Config& cfg);
std::unique_ptr<Simulator3D> create3d(const Config& cfg, std::unique_ptr<Solver3D> pressure_solver);

} // namespace SimulatorFactory

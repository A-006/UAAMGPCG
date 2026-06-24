#pragma once
#include "mesh/grid.h"
#include "mesh/bc/boundary_condition.h"
#include "io/2d/params_2d.h"
#include "util/registry.h"
#include <functional>
#include <string>

// ──────────────────────────────────────────────────────────────────
// Data-driven 2D building blocks: string-keyed registries of geometry,
// initial-condition and body-force PRIMITIVES. A scenario is then just a
// list of primitive kinds + their parameters (read from a case file via
// Param2D) — the scenario code knows nothing about specific scene names.
//
// Each registry maps a kind string (e.g. "cylinder", "vortex_dipole") to a
// builder that applies that primitive to the grid using the parameters in
// its Param2D namespace. The math itself lives in the existing helpers
// (karman.h, leapfrog math); these builders only choose and pass parameters.
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

// A geometry primitive marks solid cells (immersed obstacles).
using GeomBuilder = std::function<void(Grid&, const Config&, const Param2D&)>;

// An IC primitive seeds the staggered velocity field (additive).
using IcBuilder = std::function<void(Grid&, const Config&, const Param2D&)>;

// A body-force primitive adds a per-step force on fluid cells.
using ForceBuilder = std::function<void(Grid&, const Config&, const Param2D&)>;

// A boundary primitive returns the wall/obstacle BC stack for the run.
using BcBuilder = std::function<bc::BoundaryManager(const Config&)>;

// Registries are plain string→builder maps (no polymorphism needed), so use a
// thin wrapper around unordered_map exposed via these accessors. Each populates
// itself with the built-in primitives on first access.
const std::unordered_map<std::string, GeomBuilder>& geom_registry();
const std::unordered_map<std::string, IcBuilder>& ic_registry();
const std::unordered_map<std::string, ForceBuilder>& force_registry();
const std::unordered_map<std::string, BcBuilder>& bc_registry();

} // namespace scenarios

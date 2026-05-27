# Architecture

Layered design after the modern-CFD refactor. Each layer depends only on
those below it, and the legacy classes (`Grid`, `BoundaryConditions::*`)
remain as thin facades so existing call sites keep compiling.

```
        Simulator     ← ChorinSimulator, LFMSimulator (Algorithm 1)
            │           dispatched by SimulatorFactory (Config.dim, .time_integrator)
            ▼
   ┌───────────────────────────────────────────────┐
   │  Pressure projection  │  Advection  │  LFM   │   ← operator pipelines
   │  (uses fvc::divergence + Solver)             │
   └───────────────────────────────────────────────┘
            │
            ▼
  ┌──────────┬──────────┬──────────┬──────────────────┐
  │  fvc::   │   bc::   │  Solver  │  Fields (typed)  │
  │ operators│  patches │  + UAAMG │  CellField, etc. │
  └──────────┴──────────┴──────────┴──────────────────┘
            │            │
            ▼            ▼
        ┌───────────────────────┐
        │       Mesh2D          │   ← pure topology
        │   (nx, dx, indexing)  │
        └───────────────────────┘
            │
            ▼
        ┌───────────────────────┐
        │        Grid           │   ← Mesh2D + std::vector data
        │   (legacy facade)     │
        └───────────────────────┘
```

## Layers (bottom-up)

### `core/mesh.h` — Mesh2D
Pure topology: `nx`, `ny`, `dx`, `dy`, flat-index helpers (`iu`, `iv`, `ip`),
cell-center coordinates. No data. New code that only needs topology should
take `const Mesh2D&`.

### `core/grid.h` — Grid (legacy facade)
`class Grid : public Mesh2D` plus `std::vector<double> u, v, p` and
`std::vector<bool> solid`. Existing solver/simulator code uses Grid; it
keeps working untouched.

### `fields/field.h` — Field<Layout>
Typed wrapper around `std::vector<double>` with a layout tag
(`Cell`, `FaceX`, `FaceY`). Provides `field(i, j)` access that knows which
layout to index. Used by new operators and BCs; legacy code can also wrap
existing Grid vectors via `Field::data()`.

### `ops/operators.h` — fvc:: namespace
OpenFOAM-style finite-volume calculus as free functions on Grid:
`fvc::divergence`, `fvc::laplacian`, `fvc::vorticity`,
`fvc::kinetic_energy`. Pulled out of `Grid` (god-class) into a separate
namespace. `Grid::divergence` is kept as a back-compat shim that delegates.

### `bc/boundary_condition.h` + `bc/patches.h` — Patch-based BCs
Polymorphic `BoundaryCondition` with concrete patch types:
`InflowLeft`, `OutflowRight`, `FreeSlipTopBottom`, `NoSlipTopBottom`,
`NoSlipLeftRight`, `NoSlipImmersedSolid`. Composed via `bc::BoundaryManager`.
Scenario builders: `bc::karman(U_inf)`, `bc::smoke()`.

Legacy `BoundaryConditions::applyKarman` etc. delegate to these objects.

### `solver/` — Linear solvers (unchanged)
Jacobi / RBGS / PCG and preconditioners (GMG, AMG, UAAMG). 2D and 3D
variants currently live in separate files (`_3d` suffix) — merging them
under a single dim-aware template is a follow-up.

### `pressure/pressure.cpp` — Poisson projection
Builds RHS via `fvc::divergence`, solves via the supplied `Solver`, applies
the gradient correction. The Solver implementation is independent of
dimensionality.

### `simulator/` — Time integrators
`ChorinSimulator` (advect → diffuse → project) and `LFMSimulator`
(paper Algorithm 1: leapfrog impulse-based flow map + reinitialization).

### `simulator/factory.h` — `SimulatorFactory`
Single entry point. Dispatches on `Config.dim` and `Config.time_integrator`:

```cpp
Config cfg;
cfg.dim = 2;                       // 2 or 3 (3D Simulator wiring is a TODO)
cfg.time_integrator = "lfm";       // "chorin" | "lfm"
cfg.solver = "pcg_uaamg";

auto pressure = SimulatorFactory::make_pressure_solver(cfg);
auto sim      = SimulatorFactory::create(cfg, std::move(pressure));
```

## Directory layout

```
include/
  config/    — Config (with dim, NZ, Lz for 2D/3D switch)
  core/      — Mesh2D, Grid, Grid3D
  fields/    — Field<T, Layout> typed wrappers           (NEW)
  ops/       — fvc:: free-function operators              (NEW)
  bc/        — Patch-based BoundaryCondition objects      (NEW)
  boundary/  — Legacy BoundaryConditions:: free functions (delegates to bc::)
  solver/    — Linear solvers + preconditioners (2D & 3D)
  pressure/  — Poisson projection
  advection/ — Semi-Lagrangian utilities
  simulator/ — Simulator base + Chorin + Factory
  lfm/       — Flow-map data + LFMSimulator (paper Alg. 1)
  force/     — Cylinder Cd/Cl calculation
  io/        — VTK writer
src/
  ... mirrors include/ ...
test/
  solver/      — Standalone solver unit tests
  integration/ — End-to-end Karman / LFM validation
tools/
  run_karman_long.cpp     — LFM long run (via SimulatorFactory)
  run_karman_chorin.cpp   — Chorin reference baseline
  make_karman_*.py        — Animation generators
```

## Known follow-ups

- **Grid / Grid3D merge** — currently still separate classes. Templating
  the solver kernels on dim (or runtime-dim Grid that holds Mesh + Fields
  for any dim) would let `SimulatorFactory::create(cfg)` actually return a
  3D simulator instead of throwing.
- **Fields adoption in solvers** — solvers still take `std::vector<double>`
  directly. Migrating to `CellField` / `FaceXField` in the inner loops
  would tighten type safety with no runtime cost.
- **Configuration via YAML/JSON** — currently config is a C++ struct.
  A parser would let users drive runs without recompiling.

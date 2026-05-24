#pragma once
#include "config/config.h"
#include "core/grid.h"
#include "solver/solver.h"
#include <memory>

// Top-level fluid simulator orchestrating the time-stepping loop.
// Corresponds to OpenFOAM's applications/solvers (e.g., icoFoam).
class LFMSimulator {
public:
    LFMSimulator(const Config& cfg, std::unique_ptr<Solver> solver);

    void step();
    void run();

    const Grid& grid() const { return grid_; }
    double time() const { return t_; }
    int  step_count() const { return step_; }

private:
    Config cfg_;
    Grid   grid_;
    Grid   prev_;
    double t_ = 0;
    int    step_ = 0;
    std::unique_ptr<Solver> solver_;

    void apply_forces();
    void advect();
    void diffuse();
    void project();
    void apply_bc();
};

#pragma once
#include "config/config.h"
#include "core/grid.h"
#include "solver/solver.h"
#include "simulator/simulator_base.h"
#include <memory>

/// Standard velocity-based simulator: advect → diffuse → project (Chorin).
class ChorinSimulator : public Simulator {
public:
    ChorinSimulator(const Config& cfg, std::unique_ptr<Solver> solver);

    void step() override;
    void run();

    const Grid& grid() const override { return grid_; }
    double time() const override { return t_; }
    int  step_count() const override { return step_; }

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

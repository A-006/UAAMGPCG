#pragma once
#include "config/config.h"
#include "core/grid_3d.h"
#include "solver/solver_3d.h"
#include <memory>
#include <string>
#include <vector>

// Minimal 3D Simulator base — parallels include/simulator/simulator_base.h
// but typed on Grid3D / Solver3D.
class Simulator3D {
public:
    virtual ~Simulator3D() = default;
    virtual void step() = 0;
    virtual const Grid3D& grid() const = 0;
    virtual double time() const = 0;
    virtual int    step_count() const = 0;
};

// 3D Chorin time integrator: advect → (optional diffuse) → project.
// Initial condition is set via `set_initial(...)`; scenarios that need
// solid obstacles populate cfg_.scenario before construction or modify
// the grid externally.
class ChorinSimulator3D : public Simulator3D {
public:
    ChorinSimulator3D(const Config& cfg, std::unique_ptr<Solver3D> solver);

    void step() override;
    const Grid3D& grid() const override { return grid_; }
    double time() const override { return t_; }
    int    step_count() const override { return step_; }

    Grid3D& mutable_grid() { return grid_; }  // for scenario setup

private:
    Config  cfg_;
    Grid3D  grid_;
    Grid3D  prev_;
    double  t_ = 0;
    int     step_ = 0;
    std::unique_ptr<Solver3D> solver_;

    void apply_bc();
    void advect();
    void diffuse();
    void project();
};

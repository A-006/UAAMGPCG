#pragma once
#include "io/config.h"
#include "integrator/simulator_3d.h"
#include "mesh/bc/patches_3d.h"
#include "mesh/grid_3d.h"
#include "solver/solver_3d.h"
#include <memory>

// 3D Chorin time integrator: advect → (optional diffuse) → project.
// The 3D peer of ChorinSimulator (chorin/chorin_simulator.h).
class ChorinSimulator3D : public Simulator3D {
public:
    ChorinSimulator3D(const Config& cfg, std::unique_ptr<Solver3D> solver);

    void step() override;
    const Grid3D& grid() const override {
        return grid_;
    }
    double time() const override {
        return t_;
    }
    int step_count() const override {
        return step_;
    }

    Grid3D& mutable_grid() {
        return grid_;
    } // for IC setup

    // Replace the default BC stack (free-slip + immersed solid) with a
    // case-specific one (inflow/outflow, periodic, ...).
    void set_boundary_manager(bc::BoundaryManager3D mgr) {
        bcs_ = std::move(mgr);
    }

private:
    Config cfg_;
    Grid3D grid_;
    Grid3D prev_;
    double t_ = 0;
    int step_ = 0;
    std::unique_ptr<Solver3D> solver_;
    bc::BoundaryManager3D bcs_;

    void apply_bc();
    void advect();
    void diffuse();
    void project();
};

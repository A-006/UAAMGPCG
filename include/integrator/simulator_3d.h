#pragma once
#include "io/config.h"
#include "mesh/grid_3d.h"
#include "solver/solver_3d.h"
#include "integrator/simulation.h"
#include "mesh/bc/patches_3d.h"
#include <memory>
#include <string>
#include <vector>

// Minimal 3D Simulator base — parallels include/integrator/simulator_2d.h
// but typed on Grid3D / Solver3D.
class Simulator3D : public Simulation {
public:
    virtual void step()                = 0;
    virtual const Grid3D& grid() const = 0;
    virtual double time() const        = 0;
    virtual int step_count() const     = 0;

    // Run the 3D cycle loop (sim3d::run) to completion. Defined in runner_3d.cpp.
    void run(const Config& cfg) override;
};

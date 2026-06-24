#pragma once
#include "mesh/grid_2d.h"
#include "integrator/simulation.h"

/// Abstract 2D simulator interface. Choose implementation via Config::time_integrator.
class Simulator : public Simulation {
public:
    virtual void step()              = 0;
    virtual const Grid& grid() const = 0;
    virtual double time() const      = 0;
    virtual int step_count() const   = 0;

    // Run the 2D driver loop (sim::run) to completion. Defined in runner.cpp.
    void run(const Config& cfg) override;
};

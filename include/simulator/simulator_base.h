#pragma once
#include "core/grid.h"

/// Abstract simulator interface. Choose implementation via Config::time_integrator.
class Simulator {
public:
    virtual ~Simulator() = default;
    virtual void step() = 0;
    virtual const Grid& grid() const = 0;
    virtual double time() const = 0;
    virtual int  step_count() const = 0;
};

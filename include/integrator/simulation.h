#pragma once
#include "core/config.h"

// Common base for every simulator, 2D or 3D. The 2D (Simulator) and 3D
// (Simulator3D) hierarchies stay separate — they carry different grid types —
// but both know how to run themselves, so the launcher drives any scenario
// through one polymorphic call instead of branching on dimension.
class Simulation {
public:
    virtual ~Simulation() = default;

    // Advance the whole run to completion (drives the matching cycle loop and
    // writes output frames per cfg).
    virtual void run(const Config& cfg) = 0;
};

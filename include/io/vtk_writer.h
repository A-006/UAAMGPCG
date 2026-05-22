#pragma once
#include "core/grid.h"
#include "config/config.h"

// VTK file output for ParaView visualization.
class VtkWriter {
public:
    // Write velocity, vorticity, divergence, solid marker to VTK file
    static void write(const Grid& g, int frame, const Config& cfg);

    // Print simulation status to stdout
    static void printStatus(int step, double t, const Grid& g);
};

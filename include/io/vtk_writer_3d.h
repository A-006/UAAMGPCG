#pragma once
#include "core/grid_3d.h"
#include "config/config.h"

// 3D VTK writer — parallels include/io/vtk_writer.h.
class VtkWriter3D {
public:
    // Writes a structured-points VTK file with velocity (3-vector),
    // vorticity magnitude, divergence, and solid marker — opens cleanly
    // in ParaView.
    static void write(const Grid3D& g, int frame, const Config& cfg);

    static void printStatus(int step, double t, const Grid3D& g);
};

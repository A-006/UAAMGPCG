#pragma once
#include "lfm/grid.h"
#include "lfm/config.h"

// Write velocity, vorticity, divergence, solid marker to VTK file
void write_vtk(const Grid& g, int frame, const Config& cfg);

// Print simulation status to stdout
void print_status(int step, double t, const Grid& g);

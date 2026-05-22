#pragma once
#include "lfm/grid.h"

// Karman vortex street: left inflow U∞, right outflow ∂/∂x=0, top/bottom slip walls
void apply_bc_karman(Grid& g, double U_inf);

// Smoke: four walls no-slip
void apply_bc_smoke(Grid& g);

// No-slip on solid-adjacent faces
void apply_solid_bc(Grid& g);

// Mark cells within radius R of (cx,cy) as solid
void setup_cylinder(Grid& g, double cx, double cy, double R);

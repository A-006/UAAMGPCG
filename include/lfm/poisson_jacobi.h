#pragma once
#include "lfm/grid.h"
#include <vector>

// Jacobi iteration for Poisson equation ∇²p = rhs on MAC grid.
// Handles non-uniform dx,dy; Neumann BC at domain boundaries; solid cells.
// Modifies g.p in-place.
void jacobi_solve(Grid& g, const std::vector<double>& rhs, int max_iter);

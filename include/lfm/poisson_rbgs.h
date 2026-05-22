#pragma once
#include "lfm/grid.h"
#include <vector>

// Red-Black Gauss-Seidel smoother for MAC grid Poisson equation ∇²p = rhs.
// Converges ~2× faster than Jacobi. Modifies g.p in-place.
void rbgs_solve(Grid& g, const std::vector<double>& rhs_in, int max_iter);

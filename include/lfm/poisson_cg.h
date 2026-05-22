#pragma once
#include "lfm/grid.h"
#include <vector>

// Conjugate Gradient solver for MAC grid Poisson equation ∇²p = rhs.
// Converges in O(sqrt(κ)) iterations — far faster than Jacobi/RBGS.
// Modifies g.p in-place. Returns iteration count.
int cg_solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol);

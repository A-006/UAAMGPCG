#pragma once
#include "lfm/grid.h"
#include <vector>

// Preconditioned CG (V-Cycle multigrid preconditioner) for MAC grid.
// Near grid-independent convergence — ~10 iterations regardless of N.
// Modifies g.p in-place. Returns iteration count.
int pcg_solve(Grid& g, const std::vector<double>& rhs_in, int max_iter, double tol);

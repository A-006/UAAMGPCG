#pragma once
#include "core/grid_3d.h"
#include <vector>
#include <string>

/// Abstract 3D solver — skeleton.
class Solver3D {
public:
    virtual ~Solver3D() = default;
    virtual void solve(Grid3D& g, const std::vector<double>& rhs, int max_iter, double tol) = 0;
    virtual std::string name() const = 0;
};

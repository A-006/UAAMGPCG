#pragma once
#include "solver/poisson_solver.h"
#include <memory>
#include <string>

class SolverFactory {
public:
    static std::unique_ptr<PoissonSolver> create(const std::string& name);
};

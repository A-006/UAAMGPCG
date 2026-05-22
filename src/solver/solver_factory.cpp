#include "solver/solver_factory.h"
#include "solver/poisson_jacobi.h"
#include "solver/poisson_rbgs.h"
#include "solver/poisson_cg.h"
#include "solver/poisson_pcg.h"

std::unique_ptr<PoissonSolver> SolverFactory::create(const std::string& name) {
    if (name == "jacobi") return std::make_unique<JacobiSolver>();
    if (name == "rbgs")   return std::make_unique<RBGSSolver>();
    if (name == "cg")     return std::make_unique<CGSolver>();
    if (name == "pcg")    return std::make_unique<PCGSolver>();
    return std::make_unique<JacobiSolver>();
}

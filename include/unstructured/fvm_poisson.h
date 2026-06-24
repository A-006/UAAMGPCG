/**
 * @file fvm_poisson.h
 * @brief Cell-centred finite-volume Poisson assembly on an unstructured mesh.
 * @author liutao
 * @date 2026-06-23
 *
 * Discretises  -div(grad p) = f  with Dirichlet boundary data, using the
 * over-relaxed orthogonal two-point flux plus a deferred non-orthogonal
 * correction driven by least-squares cell gradients (MMS-verified ~2nd order on
 * triangle meshes). The matrix (orthogonal part) is symmetric and constant; the
 * non-orthogonal correction lives in the RHS and is updated each outer sweep.
 *
 * The assembled system is emitted as 0-based CSR so it can be handed straight to
 * AmgxBackend::solve_csr (or any sparse solver).
 */
#pragma once
#include "unstructured/poly_mesh.h"

#include <functional>
#include <vector>

namespace ufvm {

/// 0-based CSR linear system A x = b (columns sorted per row).
struct CSR {
    int n = 0;
    std::vector<int> row_ptr, col_idx;
    std::vector<double> vals, b;
};

using ScalarField = std::function<double(Vec2)>;

/// Boundary condition kind for a boundary face (decided from its centre).
/// Dirichlet: prescribed value (from the ScalarField). Neumann: zero normal
/// gradient dp/dn = 0 (the pressure-correction wall/inlet condition).
enum class BCType { Dirichlet, Neumann };
using BCTypeField = std::function<BCType(Vec2)>;

/// Assemble the constant orthogonal-part Laplacian matrix (A = -div grad,
/// integrated over cells). Dirichlet boundary contributes to the diagonal.
/// b is left zeroed here; fill it with build_rhs().
CSR assemble_laplacian(const PolyMesh& m, const ScalarField& dirichlet);

/// Mixed-BC overload: `bctype(Cf)` selects Dirichlet or Neumann per boundary
/// face. Dirichlet faces add gDiff to the diagonal (value -> RHS); Neumann
/// (zero-gradient) faces contribute nothing (zero flux). Requires at least one
/// Dirichlet face to pin the constant.
CSR assemble_laplacian(const PolyMesh& m, const ScalarField& dirichlet, const BCTypeField& bctype);

/// Least-squares cell gradients of p (boundary faces use the Dirichlet value).
std::vector<Vec2> ls_gradient(const PolyMesh& m, const std::vector<double>& p,
                              const ScalarField& dirichlet);

/// Mixed-BC gradient: Neumann faces use zero-gradient extrapolation (p_f=p_P).
std::vector<Vec2> ls_gradient(const PolyMesh& m, const std::vector<double>& p,
                              const ScalarField& dirichlet, const BCTypeField& bctype);

/// Build the RHS for source f + Dirichlet bc + deferred non-orthogonal
/// correction evaluated from the current cell gradients `grad`.
std::vector<double> build_rhs(const PolyMesh& m, const ScalarField& f, const ScalarField& dirichlet,
                              const std::vector<Vec2>& grad);

/// Mixed-BC RHS: Neumann faces contribute nothing (zero flux).
std::vector<double> build_rhs(const PolyMesh& m, const ScalarField& f, const ScalarField& dirichlet,
                              const std::vector<Vec2>& grad, const BCTypeField& bctype);

/// y = A x  (CSR matrix-vector product).
std::vector<double> matvec(const CSR& A, const std::vector<double>& x);

/// Simple CG (matrix is SPD) — used for solver-independent assembly checks.
std::vector<double> cg_solve(const CSR& A, const std::vector<double>& b, int max_iter, double tol);

/// Full Poisson solve with deferred non-orthogonal correction. `solver` maps a
/// CSR (with its RHS in A.b) to the solution; pass a CG- or AmgX-backed functor.
std::vector<double> solve_poisson(const PolyMesh& m, const ScalarField& f,
                                  const ScalarField& dirichlet, int n_outer,
                                  const std::function<std::vector<double>(const CSR&)>& solver);

/// Mixed-BC Poisson solve (Dirichlet/Neumann per boundary face).
std::vector<double> solve_poisson(const PolyMesh& m, const ScalarField& f,
                                  const ScalarField& dirichlet, int n_outer,
                                  const std::function<std::vector<double>(const CSR&)>& solver,
                                  const BCTypeField& bctype);

} // namespace ufvm

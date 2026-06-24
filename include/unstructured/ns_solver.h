/**
 * @file ns_solver.h
 * @brief Unsteady incompressible Navier-Stokes on an unstructured collocated
 *        FVM mesh, Chorin fractional-step projection.
 * @author liutao
 * @date 2026-06-23
 *
 * Reuses the verified FVM Poisson machinery (over-relaxed orthogonal gDiff +
 * least-squares-gradient non-orthogonal correction) for the pressure step.
 * Momentum: explicit convection (central, using divergence-free face mass
 * fluxes) + viscous diffusion (orthogonal + non-orth correction). Projection
 * removes the divergence via the constant SPD Laplacian; face mass fluxes are
 * corrected with the compact gDiff stencil (Rhie-Chow-style coupling) so the
 * collocated arrangement does not checkerboard.
 *
 * Boundary conditions are supplied as callbacks of (position, time):
 *   bc_u, bc_v  Dirichlet velocity on a face flagged Dirichlet;
 *   bc_p        Dirichlet pressure on a face flagged pressure-Dirichlet.
 * A per-face BoundaryKind classifies each boundary face.
 */
#pragma once
#include "unstructured/poly_mesh.h"

#include <functional>
#include <vector>

namespace ufvm {

enum class BKind { VelocityDirichlet, PressureDirichlet };

using SpaceTime = std::function<double(Vec2, double)>;

struct NSConfig {
    double nu    = 0.01;  ///< kinematic viscosity
    int p_outer  = 2;     ///< pressure non-orthogonal correction sweeps
    int p_iters  = 5000;  ///< pressure linear-solver max iters
    double p_tol = 1e-10; ///< pressure linear-solver tolerance
};

/// Solver state. faceFlux is the divergence-free mass flux through each face
/// (same indexing as mesh.faces), oriented owner->nb.
struct NSState {
    std::vector<double> u, v, p;
    std::vector<double> faceFlux;
};

/// Per-face boundary classification (size = number of boundary faces, indexed
/// by the position of the boundary face within mesh.faces via bface_index).
struct NSBoundary {
    std::vector<BKind> kind; ///< one entry per face (interior faces unused)
    SpaceTime bc_u, bc_v, bc_p;
};

/// Initialise state from analytic fields at t0 (and seed divergence-free-ish
/// face fluxes by interpolation).
NSState ns_init(const PolyMesh& m, const SpaceTime& u0, const SpaceTime& v0, const SpaceTime& p0,
                double t0);

/// Advance one Chorin step from time t to t+dt. Pressure solved with `solver`
/// (CG- or AmgX-backed); the assembled constant matrix `A` is passed in.
void ns_step(const PolyMesh& m, const struct CSR& A, NSState& s, const NSBoundary& bc,
             const NSConfig& cfg, double t, double dt,
             const std::function<std::vector<double>(const struct CSR&)>& solver);

} // namespace ufvm

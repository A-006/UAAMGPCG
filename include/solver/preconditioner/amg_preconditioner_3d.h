/**
 * @file amg_preconditioner_3d.h
 * @brief 3D Algebraic Multigrid (AMG) V-cycle preconditioner.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/preconditioner/preconditioner_3d.h"
#include <vector>

/**
 * @brief 3D Unsmoothed Aggregation Algebraic Multigrid preconditioner.
 *
 * Builds coarse levels by aggregating unknowns based on matrix connectivity
 * rather than geometric position. Uses the Galerkin product
 * \f$A_c = P^T A_f P\f$ to build coarse operators matrix-free.
 *
 * - Smoother: Red-Black Gauss-Seidel (7-point stencil).
 * - Aggregation: \f$2 \times 2 \times 2\f$ blocks mapped to structured coarse grid.
 * - Prolongation: piecewise-constant (each fine cell belongs to one aggregate).
 * - Restriction: \f$P^T\f$ (sum of residuals in each aggregate).
 *
 * The hierarchy is cached and rebuilt only when grid dimensions change.
 */
class AMGPreconditioner3D : public Preconditioner3D {
public:
    void apply(const Grid3D& g, const std::vector<double>& r,
               std::vector<double>& z) override;
    std::string name() const override { return "AMG3D"; }

private:
    /// One level of the 3D AMG hierarchy.
    struct AggLevel {
        int nx, ny, nz;
        double dx, dy, dz;
        double idx2, idy2, idz2, diag;   ///< Cached stencil coefficients.
        std::vector<double> p;            ///< Pressure / correction.
        std::vector<double> b;            ///< Right-hand side.
        std::vector<bool>   solid;        ///< Solid mask.
        std::vector<int>    agg;          ///< Aggregate id per fine cell (-1 = solid).
    };

    std::vector<AggLevel> levels_;                       ///< Hierarchy (finest to coarsest).
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1; ///< Cache key.

    void buildHierarchy(const Grid3D& g);
    void vCycle(int level, int nlevels);

    static void smooth           (AggLevel& L, int sweeps);
    static void smoothReverse    (AggLevel& L, int sweeps);
    static void restrictResidual (const AggLevel& fine, AggLevel& coarse);
    static void prolongateAdd    (const AggLevel& coarse, AggLevel& fine);
    static void buildAggregates  (AggLevel& fine, const AggLevel& coarse);
    static void restrictSolid    (const AggLevel& fine, AggLevel& coarse);
};

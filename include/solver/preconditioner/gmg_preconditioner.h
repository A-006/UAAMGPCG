/**
 * @file gmg_preconditioner.h
 * @brief Geometric Multigrid (GMG) V-cycle preconditioner.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "solver/preconditioner/preconditioner.h"
#include <vector>

/**
 * @brief Geometric Multigrid V-cycle preconditioner.
 *
 * Builds coarse levels by halving grid dimensions (geometric coarsening).
 * - Smoother: Red-Black Gauss-Seidel (2 pre + 2 post sweeps).
 * - Restriction: 4-to-1 averaging of the fine residual.
 * - Prolongation: piecewise-constant interpolation.
 * - Coarsest level: 20 RBGS sweeps as approximate solve.
 *
 * The hierarchy is cached and rebuilt only when grid dimensions change.
 */
class GMGPreconditioner : public Preconditioner {
public:
    void apply(const Grid& g, const std::vector<double>& r,
               std::vector<double>& z) override;
    std::string name() const override { return "GMG"; }

private:
    /// One level of the GMG hierarchy.
    struct Level {
        int nx, ny;
        double dx, dy;
        std::vector<double> p;     ///< Pressure / correction.
        std::vector<double> b;     ///< Right-hand side.
        std::vector<bool> solid;   ///< Solid mask.
    };

    std::vector<Level> levels_;           ///< Hierarchy (finest to coarsest).
    int cached_nx_ = -1, cached_ny_ = -1; ///< Cache key for the hierarchy.

    void buildHierarchy(const Grid& g);
    void vCycle(int level, int nlevels);

    static void restrictSolid     (const Level& fine, Level& coarse);
    static void smooth            (Level& L, int sweeps);
    static void smoothReverse     (Level& L, int sweeps);
    static void restrictResidual  (const Level& fine, Level& coarse);
    static void prolongateAdd     (const Level& coarse, Level& fine);
};

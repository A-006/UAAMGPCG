/**
 * @file gmg_preconditioner_3d.h
 * @brief 3D Geometric Multigrid (GMG) V-cycle preconditioner.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/preconditioner/preconditioner_3d.h"
#include <vector>

/**
 * @brief 3D Geometric Multigrid V-cycle preconditioner.
 *
 * Builds coarse levels by halving grid dimensions in all three axes.
 * - Smoother: Red-Black Gauss-Seidel (7-point stencil, 2 pre + 2 post sweeps).
 * - Restriction: 8-to-1 averaging of the fine residual (2x2x2 blocks).
 * - Prolongation: piecewise-constant injection.
 * - Coarsest level: 20 RBGS sweeps as approximate solve.
 *
 * The hierarchy is cached and rebuilt only when grid dimensions change.
 */
class GMGPreconditioner3D : public Preconditioner3D {
public:
    void apply(const Grid3D& g, const std::vector<double>& r,
               std::vector<double>& z) override;
    std::string name() const override { return "GMG3D"; }

private:
    /// One level of the 3D GMG hierarchy.
    struct Level {
        int nx, ny, nz;
        double dx, dy, dz;
        std::vector<double> p;     ///< Pressure / correction.
        std::vector<double> b;     ///< Right-hand side.
        std::vector<bool> solid;   ///< Solid mask.
    };

    std::vector<Level> levels_;                    ///< Hierarchy (finest to coarsest).
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1; ///< Cache key.

    void buildHierarchy(const Grid3D& g);
    void vCycle(int level, int nlevels);

    static void restrictSolid     (const Level& fine, Level& coarse);
    static void smooth            (Level& L, int sweeps);
    static void smoothReverse     (Level& L, int sweeps);
    static void restrictResidual  (const Level& fine, Level& coarse);
    static void prolongateAdd     (const Level& coarse, Level& fine);
};

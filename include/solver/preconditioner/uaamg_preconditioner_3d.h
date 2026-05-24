/**
 * @file uaamg_preconditioner_3d.h
 * @brief 3D UAAMG V-cycle preconditioner.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/preconditioner/preconditioner_3d.h"
#include <vector>

/**
 * @brief 3D UAAMG V-cycle preconditioner (Algorithm 3 from Sun et al. SIGGRAPH 2025).
 *
 * 1 pre + 1 post RBGS, scale-2 trilinear prolongation,
 * 8-to-1 averaging restriction, column-major indexing.
 *
 * The hierarchy is cached and rebuilt only when grid dimensions change.
 */
class UAAMGPreconditioner3D : public Preconditioner3D {
public:
    void apply(const Grid3D& g, const std::vector<double>& r,
               std::vector<double>& z) override;
    std::string name() const override { return "UAAMG3D"; }

private:
    /// One level of the 3D UAAMG hierarchy.
    struct Level {
        int nx, ny, nz;
        double dx, dy, dz;
        std::vector<double> p, b;
        std::vector<bool> solid;
    };

    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;

    void buildHierarchy(const Grid3D& g);
    void vCycle(int level, int nlevels);

    static void restrictSolid    (const Level& fine, Level& coarse);
    static void smooth           (Level& L, int sweeps);
    static void restrictResidual (const Level& fine, Level& coarse);
    static void prolongateAdd    (const Level& coarse, Level& fine);
};

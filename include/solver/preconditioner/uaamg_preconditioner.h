#pragma once
#include "solver/preconditioner/preconditioner.h"
#include <vector>

class UAAMGPreconditioner : public Preconditioner {
public:
    void apply(const Grid& g, const std::vector<double>& r,
               std::vector<double>& z) override;
    std::string name() const override { return "UAAMG"; }

private:
    struct Level {
        int nx, ny;
        double dx, dy;
        std::vector<double> p, b;
        std::vector<bool> solid;
        // Matrix-free stencil coefficients: cx[c]=coupling to +x neighbour,
        // cy analogous; diag[c]=row sum. See the 3D header for the convention.
        std::vector<double> diag, cx, cy;
    };
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1;

    void buildHierarchy(const Grid& g);
    void vCycle(int level, int nlevels);
    static void restrictSolid(const Level& fine, Level& coarse);
    static void setupFineCoeffs(Level& L);                        // finest stencil from solid
    static void galerkinCoarsen(const Level& fine, Level& coarse); // A_c = Rᵀ A P
    static void smooth(Level& L, int sweeps, bool reverse=false);  // RBGS (stored coeffs)
    static void restrictResidual(const Level& fine, Level& coarse); // R = Pᵀ (sum)
    static void prolongateAdd(const Level& coarse, Level& fine);   // x += 2 P x_c
};

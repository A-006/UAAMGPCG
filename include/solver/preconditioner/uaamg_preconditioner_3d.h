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
 * Paper-faithful matrix-free UAAMG:
 *  - Constant prolongation P (injection) with ×2 correction scaling (Eq. 11);
 *  - 8-to-1 restriction R = Pᵀ (sum over 2×2×2 children);
 *  - Galerkin coarse operator A_{l+1} = R_l A_l P_l (Eq. 12), which for the
 *    Poisson stencil stays matrix-free: the coarse +x coupling is the sum of
 *    the four fine +x couplings across the shared face, and the coarse diagonal
 *    is the sum of the six coarse couplings (Neumann zero-row-sum preserved).
 *  - RBGS smoother, V(1,1).
 *
 * Each level stores its stencil as four coefficient channels (diag, cx, cy, cz)
 * — cx[c] is the coupling between cell c and its +x neighbour, A[c,c+ex]=-cx[c];
 * the −x coupling of c is cx[c-ex]. This makes the operator fully matrix-free
 * and lets the Galerkin coarsening produce correct variable coefficients near
 * solids/boundaries.
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
        // Matrix-free stencil coefficients (per cell, sized (nx+2)(ny+2)(nz+2)):
        //   cx[c] = coupling to +x neighbour, cy/cz analogous; diag[c] = row sum.
        std::vector<double> diag, cx, cy, cz;
    };

    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;

    void buildHierarchy(const Grid3D& g);
    void vCycle(int level, int nlevels);

    static void restrictSolid    (const Level& fine, Level& coarse);
    static void setupFineCoeffs  (Level& L);                       // finest stencil from solid mask
    static void galerkinCoarsen  (const Level& fine, Level& coarse); // A_c = Rᵀ A P
    static void smooth           (Level& L, int sweeps, bool reverse=false); // RBGS (stored coeffs)
    static void restrictResidual (const Level& fine, Level& coarse); // R = Pᵀ (sum)
    static void prolongateAdd    (const Level& coarse, Level& fine); // x += 2 P x_c
};

/**
 * @file uaamg_preconditioner.cpp
 * @brief 2D UAAMG V-cycle — Algorithm 3 from Sun et al. SIGGRAPH 2025.
 *
 * Paper-faithful matrix-free UAAMG (2D analogue of the 3D version):
 *   - Constant prolongation P (injection) with ×2 correction scaling (Eq. 11);
 *   - 4-to-1 restriction R = Pᵀ (sum over 2×2 children);
 *   - Galerkin coarse operator A_{l+1} = R_l A_l P_l (Eq. 12), kept matrix-free:
 *     coarse +x coupling = sum of the 2 fine +x couplings across the shared edge,
 *     coarse diagonal = sum of the 4 coarse couplings (Neumann zero row sum);
 *   - RBGS smoother, symmetric V(1,1) (forward pre-smooth + reverse post-smooth)
 *     so the preconditioner is SPD for CG.
 *
 * Each level stores stencil channels (diag, cx, cy). cx[c] is the coupling
 * between c and its +x neighbour (A[c,c+ex] = -cx[c]); the −x coupling is cx[c-ex].
 */
#include "solver/preconditioner/uaamg_preconditioner.h"
#include <algorithm>
#include <cmath>

static inline int cidx(int i, int j, int stride) { return i + j * stride; }

void UAAMGPreconditioner::buildHierarchy(const Grid& fine) {
    levels_.clear();
    int nx = fine.nx, ny = fine.ny;
    double dx = fine.dx, dy = fine.dy;
    while (nx >= 2 && ny >= 2) {
        Level L;
        L.nx = nx; L.ny = ny; L.dx = dx; L.dy = dy;
        int N = (nx+2)*(ny+2);
        L.p.resize(N, 0.0);
        L.b.resize(N, 0.0);
        L.solid.resize(N, false);
        L.diag.resize(N, 0.0);
        L.cx.resize(N, 0.0);
        L.cy.resize(N, 0.0);
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4) break;
        nx /= 2; ny /= 2; dx *= 2.0; dy *= 2.0;
    }
}

void UAAMGPreconditioner::restrictSolid(const Level& fine, Level& coarse) {
    int fs = fine.nx + 2, cs = coarse.nx + 2;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int i_f = 2*ic - 1, j_f = 2*jc - 1, sc = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    if (fine.solid[cidx(i_f+di, j_f+dj, fs)]) sc++;
            coarse.solid[cidx(ic, jc, cs)] = (sc >= 2);
        }
}

// ── Finest-level stencil from the solid mask ──
void UAAMGPreconditioner::setupFineCoeffs(Level& L) {
    int nx = L.nx, ny = L.ny, s = nx + 2;
    double idx2 = 1.0/(L.dx*L.dx), idy2 = 1.0/(L.dy*L.dy);
    std::fill(L.cx.begin(), L.cx.end(), 0.0);
    std::fill(L.cy.begin(), L.cy.end(), 0.0);
    std::fill(L.diag.begin(), L.diag.end(), 0.0);
    for (int i=1;i<=nx;i++)
        for (int j=1;j<=ny;j++) {
            int id = cidx(i,j,s);
            if (L.solid[id]) continue;
            if (i<nx && !L.solid[cidx(i+1,j,s)]) L.cx[id] = idx2;
            if (j<ny && !L.solid[cidx(i,j+1,s)]) L.cy[id] = idy2;
        }
    for (int i=1;i<=nx;i++)
        for (int j=1;j<=ny;j++) {
            int id = cidx(i,j,s);
            if (L.solid[id]) continue;
            L.diag[id] = L.cx[id] + L.cx[cidx(i-1,j,s)] + L.cy[id] + L.cy[cidx(i,j-1,s)];
        }
}

// ── Galerkin coarse stencil: A_c = Rᵀ A P (R=Pᵀ sum, P=constant injection). ──
// Coarse +x coupling = sum of the 2 fine +x couplings across the shared edge.
void UAAMGPreconditioner::galerkinCoarsen(const Level& fine, Level& coarse) {
    int fs = fine.nx + 2, cs = coarse.nx + 2;
    std::fill(coarse.cx.begin(), coarse.cx.end(), 0.0);
    std::fill(coarse.cy.begin(), coarse.cy.end(), 0.0);
    std::fill(coarse.diag.begin(), coarse.diag.end(), 0.0);
    for (int ic=1; ic<=coarse.nx; ic++)
        for (int jc=1; jc<=coarse.ny; jc++) {
            int cid = cidx(ic,jc,cs);
            if (coarse.solid[cid]) continue;
            int i_f=2*ic-1, j_f=2*jc-1;
            double sx=0, sy=0;
            for (int dj=0; dj<2; dj++) sx += fine.cx[cidx(i_f+1, j_f+dj, fs)]; // +x edge = fine i_f+1
            for (int di=0; di<2; di++) sy += fine.cy[cidx(i_f+di, j_f+1, fs)];
            coarse.cx[cid] = sx;
            coarse.cy[cid] = sy;
        }
    for (int ic=1; ic<=coarse.nx; ic++)
        for (int jc=1; jc<=coarse.ny; jc++) {
            int cid = cidx(ic,jc,cs);
            if (coarse.solid[cid]) continue;
            coarse.diag[cid] = coarse.cx[cid] + coarse.cx[cidx(ic-1,jc,cs)]
                             + coarse.cy[cid] + coarse.cy[cidx(ic,jc-1,cs)];
        }
}

// ── RBGS smoother using stored coefficients (red-black by (i+j) parity) ──
void UAAMGPreconditioner::smooth(Level& L, int sweeps, bool reverse) {
    int nx = L.nx, ny = L.ny, s = nx + 2;
    auto sweep = [&](int parity) {
        for (int i=1;i<=nx;i++)
            for (int j=1;j<=ny;j++) {
                if (((i+j)&1)!=parity) continue;
                int id=cidx(i,j,s);
                if (L.solid[id] || L.diag[id] < 1e-30) continue;
                int im=cidx(i-1,j,s), jm=cidx(i,j-1,s);
                double nb = L.cx[id]*L.p[cidx(i+1,j,s)] + L.cx[im]*L.p[im]
                          + L.cy[id]*L.p[cidx(i,j+1,s)] + L.cy[jm]*L.p[jm];
                L.p[id] = (L.b[id] + nb) / L.diag[id];
            }
    };
    int p0 = reverse ? 0 : 1;
    for (int sw=0; sw<sweeps; sw++) { sweep(p0); sweep(1-p0); }
}

// ── Residual restriction R = Pᵀ (sum over 2×2 children) ──
void UAAMGPreconditioner::restrictResidual(const Level& fine, Level& coarse) {
    int fs = fine.nx + 2, cs = coarse.nx + 2;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int cid = cidx(ic, jc, cs);
            if (coarse.solid[cid]) continue;
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            double sum = 0;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++) {
                    int fi = i_f + di, fj = j_f + dj, fidx = cidx(fi, fj, fs);
                    if (fine.solid[fidx]) continue;
                    int im=cidx(fi-1,fj,fs), jm=cidx(fi,fj-1,fs);
                    double Ax = fine.diag[fidx]*fine.p[fidx]
                              - fine.cx[fidx]*fine.p[cidx(fi+1,fj,fs)] - fine.cx[im]*fine.p[im]
                              - fine.cy[fidx]*fine.p[cidx(fi,fj+1,fs)] - fine.cy[jm]*fine.p[jm];
                    sum += fine.b[fidx] - Ax;
                }
            coarse.b[cid] = sum;
        }
}

// ── Prolongation: constant injection with ×2 scaling (paper Eq. 11) ──
void UAAMGPreconditioner::prolongateAdd(const Level& coarse, Level& fine) {
    int cs = coarse.nx + 2, fs = fine.nx + 2;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++) {
            int cid = cidx(ic, jc, cs);
            if (coarse.solid[cid]) continue;
            double val = 2.0 * coarse.p[cid];  // ×2 improves the CG preconditioner [Stüben 2001]
            int i_f = 2*ic - 1, j_f = 2*jc - 1;
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    if (!fine.solid[cidx(i_f+di, j_f+dj, fs)])
                        fine.p[cidx(i_f+di, j_f+dj, fs)] += val;
        }
}

void UAAMGPreconditioner::vCycle(int level, int nlevels) {
    Level& L = levels_[level];
    if (level == nlevels - 1) {                  // coarsest: symmetric RBGS
        for (int s=0;s<10;s++) { smooth(L,1,false); smooth(L,1,true); }
        return;
    }
    smooth(L, 1, false);                         // pre-smooth (forward)
    Level& coarse = levels_[level + 1];
    restrictResidual(L, coarse);
    std::fill(coarse.p.begin(), coarse.p.end(), 0.0);
    vCycle(level + 1, nlevels);
    prolongateAdd(coarse, L);
    smooth(L, 1, true);                          // post-smooth (reverse → symmetric)
}

void UAAMGPreconditioner::apply(const Grid& g, const std::vector<double>& r,
                                 std::vector<double>& z) {
    if (cached_nx_ != g.nx || cached_ny_ != g.ny) {
        buildHierarchy(g); cached_nx_ = g.nx; cached_ny_ = g.ny;
    }
    int nx = g.nx, ny = g.ny, nl = (int)levels_.size(), stride0 = nx + 2;
    for (int i = 0; i <= nx+1; i++)
        for (int j = 0; j <= ny+1; j++)
            levels_[0].solid[cidx(i,j,stride0)] = g.is_solid(i,j);
    for (int l = 1; l < nl; l++) restrictSolid(levels_[l-1], levels_[l]);

    setupFineCoeffs(levels_[0]);
    for (int l = 1; l < nl; l++) galerkinCoarsen(levels_[l-1], levels_[l]);

    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            levels_[0].b[cidx(i,j,stride0)] = r[g.ip(i,j)];
    for (int l = 0; l < nl; l++) std::fill(levels_[l].p.begin(), levels_[l].p.end(), 0.0);
    vCycle(0, nl);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            z[g.ip(i,j)] = levels_[0].p[cidx(i,j,stride0)];
}

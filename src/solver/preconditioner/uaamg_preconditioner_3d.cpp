#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include <algorithm>
#include <cmath>

/// 3D column-major: i + j*(nx+2) + k*(nx+2)*(ny+2)
static inline int cidx(int i, int j, int k, int nx, int ny) {
    int si = nx + 2;
    return i + j * si + k * si * (ny + 2);
}

void UAAMGPreconditioner3D::buildHierarchy(const Grid3D& fine) {
    levels_.clear();
    int nx = fine.nx, ny = fine.ny, nz = fine.nz;
    double dx = fine.dx, dy = fine.dy, dz = fine.dz;
    while (nx >= 2 && ny >= 2 && nz >= 2) {
        Level L;
        L.nx = nx; L.ny = ny; L.nz = nz;
        L.dx = dx; L.dy = dy; L.dz = dz;
        int N = (nx+2)*(ny+2)*(nz+2);
        L.p.resize(N, 0.0);
        L.b.resize(N, 0.0);
        L.solid.resize(N, false);
        L.diag.resize(N, 0.0);
        L.cx.resize(N, 0.0);
        L.cy.resize(N, 0.0);
        L.cz.resize(N, 0.0);
        levels_.push_back(std::move(L));
        if (nx <= 4 || ny <= 4 || nz <= 4) break;
        nx /= 2; ny /= 2; nz /= 2;
        dx *= 2.0; dy *= 2.0; dz *= 2.0;
    }
}

void UAAMGPreconditioner3D::restrictSolid(const Level& fine, Level& coarse) {
    int fnx = fine.nx, fny = fine.ny;
    int cnx = coarse.nx, cny = coarse.ny;
    for (int ic = 1; ic <= coarse.nx; ic++)
        for (int jc = 1; jc <= coarse.ny; jc++)
            for (int kc = 1; kc <= coarse.nz; kc++) {
                int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1, sc=0;
                for (int di=0;di<2;di++) for(int dj=0;dj<2;dj++) for(int dk=0;dk<2;dk++)
                    if (fine.solid[cidx(i_f+di,j_f+dj,k_f+dk,fnx,fny)]) sc++;
                coarse.solid[cidx(ic,jc,kc,cnx,cny)] = (sc >= 4);
            }
}

// ── Finest-level stencil: build matrix-free coefficients from the solid mask ──
// cx[c]=idx2 when c and c+ex are both fluid (Neumann: coupling to a solid/boundary
// neighbour is dropped). diag[c] = sum of the six active couplings → zero row sum.
void UAAMGPreconditioner3D::setupFineCoeffs(Level& L) {
    int nx = L.nx, ny = L.ny, nz = L.nz;
    double idx2 = 1.0/(L.dx*L.dx), idy2 = 1.0/(L.dy*L.dy), idz2 = 1.0/(L.dz*L.dz);
    std::fill(L.cx.begin(), L.cx.end(), 0.0);
    std::fill(L.cy.begin(), L.cy.end(), 0.0);
    std::fill(L.cz.begin(), L.cz.end(), 0.0);
    std::fill(L.diag.begin(), L.diag.end(), 0.0);

    for (int i=1;i<=nx;i++)
        for (int j=1;j<=ny;j++)
            for (int k=1;k<=nz;k++) {
                int id = cidx(i,j,k,nx,ny);
                if (L.solid[id]) continue;
                if (i<nx && !L.solid[cidx(i+1,j,k,nx,ny)]) L.cx[id] = idx2;
                if (j<ny && !L.solid[cidx(i,j+1,k,nx,ny)]) L.cy[id] = idy2;
                if (k<nz && !L.solid[cidx(i,j,k+1,nx,ny)]) L.cz[id] = idz2;
            }
    for (int i=1;i<=nx;i++)
        for (int j=1;j<=ny;j++)
            for (int k=1;k<=nz;k++) {
                int id = cidx(i,j,k,nx,ny);
                if (L.solid[id]) continue;
                L.diag[id] = L.cx[id] + L.cx[cidx(i-1,j,k,nx,ny)]
                           + L.cy[id] + L.cy[cidx(i,j-1,k,nx,ny)]
                           + L.cz[id] + L.cz[cidx(i,j,k-1,nx,ny)];
            }
}

// ── Galerkin coarse stencil: A_c = Rᵀ A P (R=Pᵀ sum, P=constant injection). ──
// Coarse +x coupling = sum of the 4 fine +x couplings across the shared face.
// Coarse diagonal = sum of the 6 coarse couplings (Neumann zero row sum).
void UAAMGPreconditioner3D::galerkinCoarsen(const Level& fine, Level& coarse) {
    int fnx = fine.nx, fny = fine.ny;
    int cnx = coarse.nx, cny = coarse.ny;
    std::fill(coarse.cx.begin(), coarse.cx.end(), 0.0);
    std::fill(coarse.cy.begin(), coarse.cy.end(), 0.0);
    std::fill(coarse.cz.begin(), coarse.cz.end(), 0.0);
    std::fill(coarse.diag.begin(), coarse.diag.end(), 0.0);

    for (int ic=1; ic<=coarse.nx; ic++)
        for (int jc=1; jc<=coarse.ny; jc++)
            for (int kc=1; kc<=coarse.nz; kc++) {
                int cid = cidx(ic,jc,kc,cnx,cny);
                if (coarse.solid[cid]) continue;
                int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1;
                double sx=0, sy=0, sz=0;
                // +x face of the block is at fine i = 2*ic (= i_f+1)
                for (int dj=0; dj<2; dj++) for (int dk=0; dk<2; dk++)
                    sx += fine.cx[cidx(i_f+1, j_f+dj, k_f+dk, fnx, fny)];
                for (int di=0; di<2; di++) for (int dk=0; dk<2; dk++)
                    sy += fine.cy[cidx(i_f+di, j_f+1, k_f+dk, fnx, fny)];
                for (int di=0; di<2; di++) for (int dj=0; dj<2; dj++)
                    sz += fine.cz[cidx(i_f+di, j_f+dj, k_f+1, fnx, fny)];
                coarse.cx[cid] = sx;
                coarse.cy[cid] = sy;
                coarse.cz[cid] = sz;
            }
    for (int ic=1; ic<=coarse.nx; ic++)
        for (int jc=1; jc<=coarse.ny; jc++)
            for (int kc=1; kc<=coarse.nz; kc++) {
                int cid = cidx(ic,jc,kc,cnx,cny);
                if (coarse.solid[cid]) continue;
                coarse.diag[cid] = coarse.cx[cid] + coarse.cx[cidx(ic-1,jc,kc,cnx,cny)]
                                 + coarse.cy[cid] + coarse.cy[cidx(ic,jc-1,kc,cnx,cny)]
                                 + coarse.cz[cid] + coarse.cz[cidx(ic,jc,kc-1,cnx,cny)];
            }
}

// ── RBGS smoother using stored coefficients (red-black by (i+j+k) parity) ──
// reverse=false → forward sweep (odd, even); reverse=true → (even, odd).
// The post-smooth uses the reverse order so the V-cycle is a symmetric SPD
// operator (red-black GS forward-only is not symmetric → weak as a CG
// preconditioner). This matches the paper's measured ~14-iter performance.
void UAAMGPreconditioner3D::smooth(Level& L, int sweeps, bool reverse) {
    int nx = L.nx, ny = L.ny, nz = L.nz;
    auto sweep = [&](int parity) {
        for (int i=1;i<=nx;i++)
            for (int j=1;j<=ny;j++)
                for (int k=1;k<=nz;k++) {
                    if (((i+j+k)&1)!=parity) continue;
                    int id=cidx(i,j,k,nx,ny);
                    if (L.solid[id] || L.diag[id] < 1e-30) continue;
                    int idxm=cidx(i-1,j,k,nx,ny), idym=cidx(i,j-1,k,nx,ny), idzm=cidx(i,j,k-1,nx,ny);
                    double nb = L.cx[id]*L.p[cidx(i+1,j,k,nx,ny)] + L.cx[idxm]*L.p[idxm]
                              + L.cy[id]*L.p[cidx(i,j+1,k,nx,ny)] + L.cy[idym]*L.p[idym]
                              + L.cz[id]*L.p[cidx(i,j,k+1,nx,ny)] + L.cz[idzm]*L.p[idzm];
                    L.p[id] = (L.b[id] + nb) / L.diag[id];
                }
    };
    int p0 = reverse ? 0 : 1;
    for (int sw=0; sw<sweeps; sw++) { sweep(p0); sweep(1-p0); }
}

// ── Residual restriction: b_coarse = R r = Pᵀ r = sum over 2×2×2 children ──
void UAAMGPreconditioner3D::restrictResidual(const Level& fine, Level& coarse) {
    int fnx=fine.nx, fny=fine.ny, cnx=coarse.nx, cny=coarse.ny;
    for (int ic=1;ic<=coarse.nx;ic++)
        for (int jc=1;jc<=coarse.ny;jc++)
            for (int kc=1;kc<=coarse.nz;kc++) {
                int cid=cidx(ic,jc,kc,cnx,cny);
                if (coarse.solid[cid]) continue;
                int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1;
                double sum=0;
                for (int di=0;di<2;di++) for(int dj=0;dj<2;dj++) for(int dk=0;dk<2;dk++) {
                    int fi=i_f+di, fj=j_f+dj, fk=k_f+dk, fidx=cidx(fi,fj,fk,fnx,fny);
                    if (fine.solid[fidx]) continue;
                    int im=cidx(fi-1,fj,fk,fnx,fny), jm=cidx(fi,fj-1,fk,fnx,fny), km=cidx(fi,fj,fk-1,fnx,fny);
                    double Ax = fine.diag[fidx]*fine.p[fidx]
                              - fine.cx[fidx]*fine.p[cidx(fi+1,fj,fk,fnx,fny)] - fine.cx[im]*fine.p[im]
                              - fine.cy[fidx]*fine.p[cidx(fi,fj+1,fk,fnx,fny)] - fine.cy[jm]*fine.p[jm]
                              - fine.cz[fidx]*fine.p[cidx(fi,fj,fk+1,fnx,fny)] - fine.cz[km]*fine.p[km];
                    sum += fine.b[fidx] - Ax;
                }
                coarse.b[cid] = sum;
            }
}

// ── Prolongation: constant injection with ×2 scaling (paper Eq. 11) ──
void UAAMGPreconditioner3D::prolongateAdd(const Level& coarse, Level& fine) {
    int cnx=coarse.nx, cny=coarse.ny, fnx=fine.nx, fny=fine.ny;
    for (int ic=1;ic<=coarse.nx;ic++)
        for (int jc=1;jc<=coarse.ny;jc++)
            for (int kc=1;kc<=coarse.nz;kc++) {
                int cid=cidx(ic,jc,kc,cnx,cny);
                if(coarse.solid[cid]) continue;
                double val=2.0*coarse.p[cid];   // ×2 improves the CG preconditioner [Stüben 2001]
                int i_f=2*ic-1, j_f=2*jc-1, k_f=2*kc-1;
                for (int di=0;di<2;di++) for(int dj=0;dj<2;dj++) for(int dk=0;dk<2;dk++)
                    if(!fine.solid[cidx(i_f+di,j_f+dj,k_f+dk,fnx,fny)])
                        fine.p[cidx(i_f+di,j_f+dj,k_f+dk,fnx,fny)]+=val;
            }
}

void UAAMGPreconditioner3D::vCycle(int level, int nlevels) {
    Level& L=levels_[level];
    if(level==nlevels-1){                            // coarsest: symmetric RBGS
        for(int s=0;s<10;s++){ smooth(L,1,false); smooth(L,1,true); }
        return;
    }
    smooth(L,1,false);                               // pre-smooth (forward)
    Level& coarse=levels_[level+1];
    restrictResidual(L,coarse);
    std::fill(coarse.p.begin(),coarse.p.end(),0.0);
    vCycle(level+1,nlevels);
    prolongateAdd(coarse,L);
    smooth(L,1,true);                                // post-smooth (reverse → symmetric)
}

void UAAMGPreconditioner3D::apply(const Grid3D& g, const std::vector<double>& r,
                                   std::vector<double>& z) {
    if(cached_nx_!=g.nx||cached_ny_!=g.ny||cached_nz_!=g.nz){
        buildHierarchy(g); cached_nx_=g.nx; cached_ny_=g.ny; cached_nz_=g.nz;
    }
    int nx=g.nx, ny=g.ny, nz=g.nz, nl=(int)levels_.size();

    // Solid masks: finest from grid, then restrict down.
    for (int i=0;i<=nx+1;i++)
        for (int j=0;j<=ny+1;j++)
            for (int k=0;k<=nz+1;k++)
                levels_[0].solid[cidx(i,j,k,nx,ny)]=g.is_solid(i,j,k);
    for (int l=1;l<nl;l++) restrictSolid(levels_[l-1],levels_[l]);

    // Stencil coefficients: finest from solid, coarse via Galerkin A_c = Rᵀ A P.
    setupFineCoeffs(levels_[0]);
    for (int l=1;l<nl;l++) galerkinCoarsen(levels_[l-1],levels_[l]);

    // RHS at finest, zero all pressures.
    for (int i=1;i<=nx;i++)
        for (int j=1;j<=ny;j++)
            for (int k=1;k<=nz;k++)
                levels_[0].b[cidx(i,j,k,nx,ny)]=r[g.ip(i,j,k)];
    for (int l=0;l<nl;l++) std::fill(levels_[l].p.begin(),levels_[l].p.end(),0.0);

    vCycle(0,nl);

    for (int i=1;i<=nx;i++)
        for (int j=1;j<=ny;j++)
            for (int k=1;k<=nz;k++)
                z[g.ip(i,j,k)]=levels_[0].p[cidx(i,j,k,nx,ny)];
}

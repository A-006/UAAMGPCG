#include "bc/patches_3d.h"

namespace bc {

// ── FreeSlipAllFaces3D ──────────────────────────────────────────────
// Normal velocity = 0 on every wall; tangential ghost copies from interior
// (so ∂(tangent)/∂(normal) = 0, i.e. free-slip).
void FreeSlipAllFaces3D::apply(Grid3D& g) const {
    int nx = g.nx, ny = g.ny, nz = g.nz;

    // x-min (i=0) and x-max (i=nx): u=0 on the wall; v/w ghost copies inside.
    for (int k = 0; k <= nz + 1; k++) {
        for (int j = 0; j <= ny + 1; j++) {
            g.u_at(0,  Mesh3D::clamp(j, 1, ny), Mesh3D::clamp(k, 1, nz)) = 0.0;
            g.u_at(nx, Mesh3D::clamp(j, 1, ny), Mesh3D::clamp(k, 1, nz)) = 0.0;
        }
    }
    for (int k = 1; k <= nz; k++) {
        for (int j = 0; j <= ny; j++) {
            g.v_at(0,      j, k) = g.v_at(1,  j, k);
            g.v_at(nx + 1, j, k) = g.v_at(nx, j, k);
        }
        for (int j = 1; j <= ny; j++) {
            g.w_at(0,      j, k) = g.w_at(1,  j, k);
            g.w_at(nx + 1, j, k) = g.w_at(nx, j, k);
        }
    }

    // y-min (j=0) and y-max (j=ny)
    for (int k = 1; k <= nz; k++) {
        for (int i = 1; i <= nx; i++) {
            g.v_at(i, 0,  k) = 0.0;
            g.v_at(i, ny, k) = 0.0;
        }
        for (int i = 0; i <= nx; i++) {
            g.u_at(i, 0,      k) = g.u_at(i, 1,  k);
            g.u_at(i, ny + 1, k) = g.u_at(i, ny, k);
        }
        for (int i = 1; i <= nx; i++) {
            g.w_at(i, 0,      k) = g.w_at(i, 1,  k);
            g.w_at(i, ny + 1, k) = g.w_at(i, ny, k);
        }
    }

    // z-min (k=0) and z-max (k=nz)
    for (int j = 1; j <= ny; j++) {
        for (int i = 1; i <= nx; i++) {
            g.w_at(i, j, 0)  = 0.0;
            g.w_at(i, j, nz) = 0.0;
        }
        for (int i = 0; i <= nx; i++) {
            g.u_at(i, j, 0)      = g.u_at(i, j, 1);
            g.u_at(i, j, nz + 1) = g.u_at(i, j, nz);
        }
        for (int i = 1; i <= nx; i++) {
            g.v_at(i, j, 0)      = g.v_at(i, j, 1);
            g.v_at(i, j, nz + 1) = g.v_at(i, j, nz);
        }
    }
}

// ── Periodic3D ──────────────────────────────────────────────────────
// Copies the "wrap-around" interior face into the ghost layer.
void Periodic3D::apply(Grid3D& g) const {
    int nx = g.nx, ny = g.ny, nz = g.nz;

    // x: u_at(0, j, k) ↔ u_at(nx, j, k)
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= ny; j++) {
            g.u_at(0,  j, k) = g.u_at(nx,     j, k);
            g.u_at(nx, j, k) = g.u_at(0,      j, k);
            g.v_at(0,      j, k) = g.v_at(nx, j, k);
            g.v_at(nx + 1, j, k) = g.v_at(1,  j, k);
            g.w_at(0,      j, k) = g.w_at(nx, j, k);
            g.w_at(nx + 1, j, k) = g.w_at(1,  j, k);
        }
    }

    // y
    for (int k = 1; k <= nz; k++) {
        for (int i = 1; i <= nx; i++) {
            g.v_at(i, 0,  k) = g.v_at(i, ny, k);
            g.v_at(i, ny, k) = g.v_at(i, 0,  k);
            g.u_at(i, 0,      k) = g.u_at(i, ny, k);
            g.u_at(i, ny + 1, k) = g.u_at(i, 1,  k);
            g.w_at(i, 0,      k) = g.w_at(i, ny, k);
            g.w_at(i, ny + 1, k) = g.w_at(i, 1,  k);
        }
    }

    // z
    for (int j = 1; j <= ny; j++) {
        for (int i = 1; i <= nx; i++) {
            g.w_at(i, j, 0)  = g.w_at(i, j, nz);
            g.w_at(i, j, nz) = g.w_at(i, j, 0);
            g.u_at(i, j, 0)      = g.u_at(i, j, nz);
            g.u_at(i, j, nz + 1) = g.u_at(i, j, 1);
            g.v_at(i, j, 0)      = g.v_at(i, j, nz);
            g.v_at(i, j, nz + 1) = g.v_at(i, j, 1);
        }
    }
}

// ── NoSlipImmersedSolid3D ───────────────────────────────────────────
void NoSlipImmersedSolid3D::apply(Grid3D& g) const {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= ny; j++) {
            for (int i = 1; i <= nx; i++) {
                if (!g.is_solid(i, j, k)) continue;
                if (i > 1  && !g.is_solid(i - 1, j, k)) g.u_at(i - 1, j, k) = 0.0;
                if (i < nx && !g.is_solid(i + 1, j, k)) g.u_at(i,     j, k) = 0.0;
                if (j > 1  && !g.is_solid(i, j - 1, k)) g.v_at(i, j - 1, k) = 0.0;
                if (j < ny && !g.is_solid(i, j + 1, k)) g.v_at(i, j,     k) = 0.0;
                if (k > 1  && !g.is_solid(i, j, k - 1)) g.w_at(i, j, k - 1) = 0.0;
                if (k < nz && !g.is_solid(i, j, k + 1)) g.w_at(i, j, k)     = 0.0;
            }
        }
    }
}

BoundaryManager3D free_slip_box() {
    BoundaryManager3D mgr;
    mgr.add(std::make_unique<FreeSlipAllFaces3D>());
    mgr.add(std::make_unique<NoSlipImmersedSolid3D>());
    return mgr;
}

BoundaryManager3D periodic_box() {
    BoundaryManager3D mgr;
    mgr.add(std::make_unique<Periodic3D>());
    return mgr;
}

}  // namespace bc

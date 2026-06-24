// Unit tests for 3D patch-based boundary conditions.
// Real headers: core/bc/patches_3d.h, core/grid_3d.h.
#include "../test_utils.h"
#include "core/bc/patches_3d.h"
#include "core/grid_3d.h"

// Fill every MAC face with a distinct nonzero sentinel so we can detect
// exactly which faces a BC touches.
static void fill_uniform(Grid3D& g, double uval, double vval, double wval) {
    for (auto& x : g.u) x = uval;
    for (auto& x : g.v) x = vval;
    for (auto& x : g.w) x = wval;
}

int main() {
    test_header("3D Boundary Condition Patches (patches_3d)");

    // ─────────────────────────────────────────────────────────────
    // FreeSlipAllFaces3D
    // ─────────────────────────────────────────────────────────────
    {
        Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
        fill_uniform(g, 7.0, 9.0, 11.0);
        bc::FreeSlipAllFaces3D bc;
        bc.apply(g);
        int nx = g.nx, ny = g.ny, nz = g.nz;

        // Normal velocity zeroed on the six walls.
        check_approx(g.u_at(0, 2, 2), 0.0, 1e-12, "free-slip: u=0 at x-min wall (i=0)");
        check_approx(g.u_at(nx, 2, 2), 0.0, 1e-12, "free-slip: u=0 at x-max wall (i=nx)");
        check_approx(g.v_at(2, 0, 2), 0.0, 1e-12, "free-slip: v=0 at y-min wall (j=0)");
        check_approx(g.v_at(2, ny, 2), 0.0, 1e-12, "free-slip: v=0 at y-max wall (j=ny)");
        check_approx(g.w_at(2, 2, 0), 0.0, 1e-12, "free-slip: w=0 at z-min wall (k=0)");
        check_approx(g.w_at(2, 2, nz), 0.0, 1e-12, "free-slip: w=0 at z-max wall (k=nz)");

        // Interior normal faces untouched.
        check_approx(g.u_at(2, 2, 2), 7.0, 1e-12, "free-slip: interior u unchanged");
        check_approx(g.v_at(2, 2, 2), 9.0, 1e-12, "free-slip: interior v unchanged");
        check_approx(g.w_at(2, 2, 2), 11.0, 1e-12, "free-slip: interior w unchanged");

        // Tangential ghost copies (zero-gradient) — set distinct interior values.
        g.u_at(2, 1, 2) = 3.5;          // first interior u-row in y
        g.u_at(2, ny, 2) = 4.5;         // last interior u-row in y
        g.w_at(1, 2, 2) = 5.5;          // first interior w in x
        g.w_at(nx, 2, 2) = 6.5;         // last interior w in x
        bc.apply(g);
        check_approx(g.u_at(2, 0, 2), 3.5, 1e-12, "free-slip: u ghost at y-min copies interior");
        check_approx(g.u_at(2, ny + 1, 2), 4.5, 1e-12, "free-slip: u ghost at y-max copies interior");
        check_approx(g.w_at(0, 2, 2), 5.5, 1e-12, "free-slip: w ghost at x-min copies interior");
        check_approx(g.w_at(nx + 1, 2, 2), 6.5, 1e-12, "free-slip: w ghost at x-max copies interior");
    }

    // ─────────────────────────────────────────────────────────────
    // Periodic3D — ghost layers wrap around.
    // ─────────────────────────────────────────────────────────────
    {
        Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
        fill_uniform(g, 0.0, 0.0, 0.0);
        int nx = g.nx, ny = g.ny, nz = g.nz;

        // x-axis wrap for v: ghost i=0 <- interior i=nx ; ghost i=nx+1 <- i=1
        g.v_at(nx, 2, 2) = 2.2;
        g.v_at(1, 2, 2) = 3.3;
        // y-axis wrap for u: ghost j=0 <- j=ny ; ghost j=ny+1 <- j=1
        g.u_at(2, ny, 2) = 4.4;
        g.u_at(2, 1, 2) = 5.5;
        // z-axis wrap for v: ghost k=0 <- k=nz ; ghost k=nz+1 <- k=1
        g.v_at(2, 2, nz) = 6.6;
        g.v_at(2, 2, 1) = 7.7;

        bc::Periodic3D bc;
        bc.apply(g);

        check_approx(g.v_at(0, 2, 2), 2.2, 1e-12, "periodic: v ghost x-min wraps from i=nx");
        check_approx(g.v_at(nx + 1, 2, 2), 3.3, 1e-12, "periodic: v ghost x-max wraps from i=1");
        check_approx(g.u_at(2, 0, 2), 4.4, 1e-12, "periodic: u ghost y-min wraps from j=ny");
        check_approx(g.u_at(2, ny + 1, 2), 5.5, 1e-12, "periodic: u ghost y-max wraps from j=1");
        check_approx(g.v_at(2, 2, 0), 6.6, 1e-12, "periodic: v ghost z-min wraps from k=nz");
        check_approx(g.v_at(2, 2, nz + 1), 7.7, 1e-12, "periodic: v ghost z-max wraps from k=1");
    }

    // ─────────────────────────────────────────────────────────────
    // NoSlipImmersedSolid3D — every fluid-solid face zeroed.
    // This is the 3D analogue of the 2D solid-face correctness fix:
    // an interior solid cell surrounded by fluid must get ALL SIX of
    // its bounding faces driven to zero.
    // ─────────────────────────────────────────────────────────────
    {
        Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
        fill_uniform(g, 1.0, 1.0, 1.0);
        // Solid cell strictly interior (neighbors on all 6 sides are fluid).
        const int si = 2, sj = 2, sk = 2;
        g.set_solid(si, sj, sk);
        check(g.is_solid(si, sj, sk), "solid: cell flagged solid");

        bc::NoSlipImmersedSolid3D bc;
        bc.apply(g);

        // All six bounding faces of the solid cell must be zero.
        check_approx(g.u_at(si - 1, sj, sk), 0.0, 1e-12, "solid: -x face (u_at(i-1)) zeroed");
        check_approx(g.u_at(si, sj, sk), 0.0, 1e-12, "solid: +x face (u_at(i)) zeroed");
        check_approx(g.v_at(si, sj - 1, sk), 0.0, 1e-12, "solid: -y face (v_at(j-1)) zeroed");
        check_approx(g.v_at(si, sj, sk), 0.0, 1e-12, "solid: +y face (v_at(j)) zeroed");
        check_approx(g.w_at(si, sj, sk - 1), 0.0, 1e-12, "solid: -z face (w_at(k-1)) zeroed");
        check_approx(g.w_at(si, sj, sk), 0.0, 1e-12, "solid: +z face (w_at(k)) zeroed");

        // Faces NOT bounding the solid cell are untouched (no over-zeroing).
        check_approx(g.u_at(si + 1, sj, sk), 1.0, 1e-12, "solid: far +x u-face untouched");
        check_approx(g.v_at(si, sj + 1, sk), 1.0, 1e-12, "solid: far +y v-face untouched");
        check_approx(g.w_at(si, sj, sk + 1), 1.0, 1e-12, "solid: far +z w-face untouched");
    }

    // ─────────────────────────────────────────────────────────────
    // NoSlipImmersedSolid3D — solid-solid faces are NOT zeroed.
    // Two adjacent solid cells: the shared interface face stays as-is;
    // only the fluid-facing faces get zeroed.
    // ─────────────────────────────────────────────────────────────
    {
        Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
        fill_uniform(g, 1.0, 1.0, 1.0);
        // Two solids adjacent along x: (2,2,2) and (3,2,2).
        g.set_solid(2, 2, 2);
        g.set_solid(3, 2, 2);

        bc::NoSlipImmersedSolid3D bc;
        bc.apply(g);

        // Shared face between the two solids (u_at(2,2,2)) is solid-solid -> unchanged.
        check_approx(g.u_at(2, 2, 2), 1.0, 1e-12, "solid-solid: shared interface face NOT zeroed");
        // Outer fluid faces of the solid block are zeroed.
        check_approx(g.u_at(1, 2, 2), 0.0, 1e-12, "solid-solid: -x outer face zeroed");
        check_approx(g.u_at(3, 2, 2), 0.0, 1e-12, "solid-solid: +x outer face zeroed");
    }

    // ─────────────────────────────────────────────────────────────
    // NoSlipImmersedSolid3D — corner solid cell at (1,1,1).
    // Domain-edge guards (i>1, j>1, k>1, i<nx ...) mean the low-side
    // faces toward the boundary are skipped; high-side fluid faces zeroed.
    // ─────────────────────────────────────────────────────────────
    {
        Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
        fill_uniform(g, 1.0, 1.0, 1.0);
        g.set_solid(1, 1, 1);
        bc::NoSlipImmersedSolid3D bc;
        bc.apply(g);
        int nx = g.nx, ny = g.ny, nz = g.nz;
        // i<nx etc. true -> high-side fluid faces zeroed.
        check_approx(g.u_at(1, 1, 1), 0.0, 1e-12, "corner solid: +x fluid face zeroed");
        check_approx(g.v_at(1, 1, 1), 0.0, 1e-12, "corner solid: +y fluid face zeroed");
        check_approx(g.w_at(1, 1, 1), 0.0, 1e-12, "corner solid: +z fluid face zeroed");
        (void)nx; (void)ny; (void)nz;
    }

    // ─────────────────────────────────────────────────────────────
    // free_slip_box builder: composes free-slip walls + immersed solid.
    // ─────────────────────────────────────────────────────────────
    {
        bc::BoundaryManager3D mgr = bc::free_slip_box();
        check(mgr.size() == 2, "free_slip_box: contains 2 BCs (walls + solid)");
        check(!mgr.empty(), "free_slip_box: not empty");

        Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
        fill_uniform(g, 5.0, 5.0, 5.0);
        g.set_solid(2, 2, 2);
        mgr.apply(g);
        // Wall normal zeroed AND solid face zeroed in one apply().
        check_approx(g.u_at(0, 2, 2), 0.0, 1e-12, "free_slip_box: wall u=0 applied");
        check_approx(g.u_at(2, 2, 2), 0.0, 1e-12, "free_slip_box: solid face zeroed applied");
    }

    // ─────────────────────────────────────────────────────────────
    // periodic_box builder.
    // ─────────────────────────────────────────────────────────────
    {
        bc::BoundaryManager3D mgr = bc::periodic_box();
        check(mgr.size() == 1, "periodic_box: contains 1 BC");
    }

    return test_summary();
}

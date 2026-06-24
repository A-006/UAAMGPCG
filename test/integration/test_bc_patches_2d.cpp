// Unit tests for the 2D patch boundary conditions (excluding NoSlipImmersedSolid,
// which is covered by test_bc_solid.cpp).
//
// Covers: InflowLeft, OutflowRight, FreeSlipTopBottom, FreeSlipLeftRight, and the
// scenario builders free_slip_walls() and karman(U_inf). Each test fills a Grid
// with a known velocity field, applies the BC, and asserts the exact resulting
// MAC face values, matching the semantics in src/core/bc/patches.cpp.
#include "../test_utils.h"
#include "mesh/grid.h"
#include "mesh/bc/patches.h"

using bc::FreeSlipLeftRight;
using bc::FreeSlipTopBottom;
using bc::InflowLeft;
using bc::OutflowRight;

// fill every storage slot of u and v (incl. ghosts) with a unique-ish value so we
// can verify exactly which entries the BC touches.
static void fill_all(Grid& g, double uval, double vval) {
    for (auto& x : g.u)
        x = uval;
    for (auto& x : g.v)
        x = vval;
}

int main() {
    test_header("2D patch boundary conditions");

    // ── InflowLeft: u(0,j)=U_inf, v(0,j)=0 for j=1..ny, plus corners v(0,0)=v(0,ny)=0 ──
    {
        Grid g(6, 5, 1.0, 1.0);
        fill_all(g, 7.0, 3.0);
        InflowLeft inflow(2.5);
        inflow.apply(g);

        bool u_ok = true, v_ok = true;
        for (int j = 1; j <= g.ny; j++) {
            if (g.u_at(0, j) != 2.5)
                u_ok = false;
            if (g.v_at(0, j) != 0.0)
                v_ok = false;
        }
        check(u_ok, "InflowLeft: left u-faces set to U_inf");
        check(v_ok, "InflowLeft: left v-faces set to 0");
        check(g.v_at(0, 0) == 0.0 && g.v_at(0, g.ny) == 0.0,
              "InflowLeft: left corner v-ghosts set to 0");
        // interior is untouched
        check(g.u_at(3, 3) == 7.0 && g.v_at(3, 3) == 3.0, "InflowLeft: interior untouched");
        // u at i=1 (one column in) is NOT overwritten
        check(g.u_at(1, 2) == 7.0, "InflowLeft: only column i=0 of u modified");
    }

    // ── InflowLeft with negative U_inf ──
    {
        Grid g(4, 4, 2.0, 2.0);
        fill_all(g, 1.0, 1.0);
        InflowLeft inflow(-3.0);
        inflow.apply(g);
        check(g.u_at(0, 1) == -3.0 && g.u_at(0, g.ny) == -3.0,
              "InflowLeft: negative U_inf applied to all left u-faces");
    }

    // ── OutflowRight: zero-gradient. u(nx,j)=u(nx-1,j), v(nx+1,j)=v(nx,j) ──
    {
        Grid g(6, 5, 1.0, 1.0);
        fill_all(g, 0.0, 0.0);
        // set distinct interior values so the copy is observable
        for (int j = 1; j <= g.ny; j++) {
            g.u_at(g.nx - 1, j) = 1.0 + j;     // source for u(nx,j)
            g.v_at(g.nx, j)     = 10.0 + j;    // source for v(nx+1,j)
            g.u_at(g.nx, j)     = -99.0;       // garbage to be overwritten
            g.v_at(g.nx + 1, j) = -99.0;       // garbage to be overwritten
        }
        OutflowRight outflow;
        outflow.apply(g);

        bool u_ok = true;
        for (int j = 1; j <= g.ny; j++) {
            if (g.u_at(g.nx, j) != 1.0 + j)
                u_ok = false;
        }
        check(u_ok, "OutflowRight: u(nx,j) copies u(nx-1,j) (zero-gradient)");
        // v ghosts copy interior for j=1..ny-1; the j=ny corner is overwritten to 0
        // afterwards by the corner-fix lines, so check only the interior rows here.
        bool v_ok = true;
        for (int j = 1; j <= g.ny - 1; j++) {
            if (g.v_at(g.nx + 1, j) != 10.0 + j)
                v_ok = false;
        }
        check(v_ok, "OutflowRight: v(nx+1,j) copies v(nx,j) (zero-gradient) for interior rows");
        // corner rows j=0 and j=ny are explicitly zeroed (the corner overwrite wins
        // over the loop copy at j=ny).
        check(g.v_at(g.nx + 1, 0) == 0.0 && g.v_at(g.nx + 1, g.ny) == 0.0,
              "OutflowRight: right corner v-ghosts set to 0 (overrides loop at j=ny)");
    }

    // ── FreeSlipTopBottom: ghost u rows copy interior (∂u/∂y=0); v=0 on walls ──
    {
        Grid g(5, 6, 1.0, 1.0);
        fill_all(g, 0.0, 4.0);
        // distinct interior u rows so the ghost copy is observable
        for (int i = 0; i <= g.nx; i++) {
            g.u_at(i, 1)      = 100.0 + i; // source for bottom ghost u(i,0)
            g.u_at(i, g.ny)   = 200.0 + i; // source for top ghost u(i,ny+1)
            g.u_at(i, 0)      = -1.0;      // garbage
            g.u_at(i, g.ny + 1) = -1.0;    // garbage
        }
        FreeSlipTopBottom fs;
        fs.apply(g);

        bool ughost_ok = true;
        for (int i = 0; i <= g.nx; i++) {
            if (g.u_at(i, 0) != 100.0 + i)
                ughost_ok = false;
            if (g.u_at(i, g.ny + 1) != 200.0 + i)
                ughost_ok = false;
        }
        check(ughost_ok, "FreeSlipTopBottom: ghost u rows copy interior (zero normal-gradient)");

        bool vwall_ok = true;
        for (int i = 1; i <= g.nx; i++) {
            if (g.v_at(i, 0) != 0.0 || g.v_at(i, g.ny) != 0.0)
                vwall_ok = false;
        }
        check(vwall_ok, "FreeSlipTopBottom: wall-normal v zeroed on top & bottom walls");
        // interior v away from the walls untouched
        check(g.v_at(3, 3) == 4.0, "FreeSlipTopBottom: interior v untouched");
    }

    // ── FreeSlipLeftRight: u=0 on walls; ghost v columns copy interior (∂v/∂x=0) ──
    {
        Grid g(6, 5, 1.0, 1.0);
        fill_all(g, 8.0, 0.0);
        for (int j = 0; j <= g.ny; j++) {
            g.v_at(1, j)      = 300.0 + j; // source for left ghost v(0,j)
            g.v_at(g.nx, j)   = 400.0 + j; // source for right ghost v(nx+1,j)
            g.v_at(0, j)      = -1.0;      // garbage
            g.v_at(g.nx + 1, j) = -1.0;    // garbage
        }
        FreeSlipLeftRight fs;
        fs.apply(g);

        bool uwall_ok = true;
        for (int j = 1; j <= g.ny; j++) {
            if (g.u_at(0, j) != 0.0 || g.u_at(g.nx, j) != 0.0)
                uwall_ok = false;
        }
        check(uwall_ok, "FreeSlipLeftRight: wall-normal u zeroed on left & right walls");

        bool vghost_ok = true;
        for (int j = 0; j <= g.ny; j++) {
            if (g.v_at(0, j) != 300.0 + j)
                vghost_ok = false;
            if (g.v_at(g.nx + 1, j) != 400.0 + j)
                vghost_ok = false;
        }
        check(vghost_ok, "FreeSlipLeftRight: ghost v columns copy interior (zero normal-gradient)");
        // interior u away from the walls untouched
        check(g.u_at(3, 3) == 8.0, "FreeSlipLeftRight: interior u untouched");
    }

    // ── Builder free_slip_walls(): FreeSlipLeftRight + FreeSlipTopBottom + solid ──
    {
        bc::BoundaryManager mgr = bc::free_slip_walls();
        check(mgr.size() == 3, "free_slip_walls(): composes 3 BCs");
        check(!mgr.empty(), "free_slip_walls(): not empty");

        Grid g(6, 6, 1.0, 1.0);
        fill_all(g, 5.0, 5.0);
        mgr.apply(g);
        // both free-slip walls zero their wall-normal component
        check(g.u_at(0, 3) == 0.0 && g.u_at(g.nx, 3) == 0.0,
              "free_slip_walls(): left/right walls have u=0");
        check(g.v_at(3, 0) == 0.0 && g.v_at(3, g.ny) == 0.0,
              "free_slip_walls(): top/bottom walls have v=0");
    }

    // ── Builder karman(U_inf): InflowLeft + OutflowRight + FreeSlipTopBottom + solid ──
    {
        bc::BoundaryManager mgr = bc::karman(1.5);
        check(mgr.size() == 4, "karman(): composes 4 BCs");

        Grid g(8, 6, 2.0, 1.0);
        fill_all(g, 0.0, 0.0);
        // set the column just inside the right boundary so outflow copy is observable
        for (int j = 1; j <= g.ny; j++)
            g.u_at(g.nx - 1, j) = 0.9;
        mgr.apply(g);

        // inflow on the left
        bool inflow_ok = true;
        for (int j = 1; j <= g.ny; j++)
            if (g.u_at(0, j) != 1.5)
                inflow_ok = false;
        check(inflow_ok, "karman(): inflow sets left u-faces to U_inf");

        // zero-gradient outflow on the right
        bool outflow_ok = true;
        for (int j = 1; j <= g.ny; j++)
            if (g.u_at(g.nx, j) != 0.9)
                outflow_ok = false;
        check(outflow_ok, "karman(): outflow copies last interior u-column to right face");

        // free-slip top/bottom walls: v=0
        check(g.v_at(4, 0) == 0.0 && g.v_at(4, g.ny) == 0.0,
              "karman(): free-slip top/bottom walls have v=0");
    }

    // ── Idempotence: applying free-slip twice yields the same result ──
    {
        Grid g(5, 5, 1.0, 1.0);
        fill_all(g, 2.0, 3.0);
        FreeSlipTopBottom fs;
        fs.apply(g);
        std::vector<double> u_after = g.u;
        std::vector<double> v_after = g.v;
        fs.apply(g);
        check(g.u == u_after && g.v == v_after, "FreeSlipTopBottom: idempotent under repeated apply");
    }

    return test_summary();
}

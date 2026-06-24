// Regression test for the immersed-solid velocity BC (NoSlipImmersedSolid).
//
// Bug (the 2D twin of the delta-wing solid-BC gap): the old apply() guarded every
// face with `!is_solid(neighbor)`, so it zeroed ONLY fluid-facing interface faces
// and left solid<->solid interior faces (and the inflow-side face) at the
// freestream value. The fix zeroes ALL four MAC faces of every solid cell,
// matching the author's SetBcByPhiKernel.
//
// These tests pin the fixed behaviour so it cannot regress.
#include "../test_utils.h"
#include "mesh/grid.h"
#include "mesh/bc/patches.h"

#include <cmath>

using bc::NoSlipImmersedSolid;

// set every u and v MAC face to a uniform "freestream" value
static void fill_faces(Grid& g, double val) {
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            g.u_at(i, j) = val;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++)
            g.v_at(i, j) = val;
}

// max |face| over the four MAC faces of cell (i,j)
static double max_face_abs(const Grid& g, int i, int j) {
    return std::max(std::max(std::abs(g.u_at(i - 1, j)), std::abs(g.u_at(i, j))),
                    std::max(std::abs(g.v_at(i, j - 1)), std::abs(g.v_at(i, j))));
}

int main() {
    test_header("Immersed-solid velocity BC (NoSlipImmersedSolid)");
    NoSlipImmersedSolid bc;

    // ── Test 1: a 3x3 solid block — every face of every solid cell must be 0,
    //            including the solid<->solid interior faces (the regression). ──
    {
        Grid g(12, 12, 1.0, 1.0);
        fill_faces(g, 1.0);
        for (int i = 3; i <= 5; i++)
            for (int j = 3; j <= 5; j++)
                g.set_solid(i, j);

        bc.apply(g);

        double worst = 0.0;
        for (int i = 3; i <= 5; i++)
            for (int j = 3; j <= 5; j++)
                worst = std::max(worst, max_face_abs(g, i, j));
        check(worst == 0.0, "all 4 faces of every solid cell are exactly 0");

        // The center cell (4,4) is fully surrounded by solid — its faces are all
        // solid<->solid interior faces that the OLD buggy code left at 1.0.
        check(g.u_at(3, 4) == 0.0 && g.u_at(4, 4) == 0.0, "interior solid<->solid u-faces zeroed (regression)");
        check(g.v_at(4, 3) == 0.0 && g.v_at(4, 4) == 0.0, "interior solid<->solid v-faces zeroed (regression)");
    }

    // ── Test 2: faces with no solid neighbour are untouched (no over-zeroing). ──
    {
        Grid g(12, 12, 1.0, 1.0);
        fill_faces(g, 1.0);
        for (int i = 3; i <= 5; i++)
            for (int j = 3; j <= 5; j++)
                g.set_solid(i, j);

        bc.apply(g);

        // u-face at (9,9) and v-face at (9,9) are far from the block -> still 1.0
        check(g.u_at(9, 9) == 1.0 && g.v_at(9, 9) == 1.0, "far-field fluid faces unchanged");
        // a fluid<->fluid face adjacent to the block but not shared with a solid
        // cell (e.g. u(1,4), two cells left of the block) is untouched
        check(g.u_at(1, 4) == 1.0, "fluid faces not adjacent to a solid stay freestream");
    }

    // ── Test 3: a single isolated solid cell — exactly its 4 faces zeroed. ──
    {
        Grid g(10, 10, 1.0, 1.0);
        fill_faces(g, 2.5);
        g.set_solid(7, 7);

        bc.apply(g);

        check(max_face_abs(g, 7, 7) == 0.0, "isolated solid cell: all 4 faces zeroed");
        // the opposite faces of the 4 fluid neighbours are untouched
        check(g.u_at(8, 7) == 2.5, "neighbour cell's far u-face untouched");
        check(g.v_at(7, 8) == 2.5, "neighbour cell's far v-face untouched");
    }

    // ── Test 4: no solid anywhere -> field is completely unchanged. ──
    {
        Grid g(8, 8, 1.0, 1.0);
        fill_faces(g, 1.0);
        bc.apply(g);
        double worst = 0.0;
        for (int i = 0; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                worst = std::max(worst, std::abs(g.u_at(i, j) - 1.0));
        check(worst == 0.0, "no-solid grid: velocity field untouched");
    }

    return test_summary();
}

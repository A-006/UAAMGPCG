#include "boundary/boundary.h"
#include "bc/patches.h"
#include <cmath>

// Legacy free-function API kept as thin wrapper around the new bc::* objects.
// New code should prefer `bc::karman(...)` / `bc::smoke()` and apply the
// returned BoundaryManager directly.

void BoundaryConditions::applyKarman(Grid& g, double U_inf) {
    bc::InflowLeft        in(U_inf);  in.apply(g);
    bc::OutflowRight      out;        out.apply(g);
    bc::FreeSlipTopBottom slip;       slip.apply(g);
}

void BoundaryConditions::applySmoke(Grid& g) {
    bc::NoSlipLeftRight  lr;  lr.apply(g);
    bc::NoSlipTopBottom  tb;  tb.apply(g);
}

void BoundaryConditions::applySolid(Grid& g) {
    bc::NoSlipImmersedSolid solid;
    solid.apply(g);
}

void BoundaryConditions::setupCylinder(Grid& g, double cx, double cy, double R) {
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            double xc = (i - 0.5) * g.dx;
            double yc = (j - 0.5) * g.dy;
            if ((xc - cx) * (xc - cx) + (yc - cy) * (yc - cy) < R * R)
                g.set_solid(i, j);
        }
    }
}

#include "scenarios/fire_ball.h"
#include <cmath>

namespace scenarios {

void seed_fire_ball(ScalarField3D& T, const Grid3D& g, const FireBall& fb) {
    T.fill(fb.T_ref);
    double r2 = fb.radius * fb.radius;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                double x = (i - 0.5) * g.dx;
                double y = (j - 0.5) * g.dy;
                double z = (k - 0.5) * g.dz;
                double rx = x - fb.center[0];
                double ry = y - fb.center[1];
                double rz = z - fb.center[2];
                double d2 = rx*rx + ry*ry + rz*rz;
                T(i, j, k) = fb.T_ref + (fb.T_hot - fb.T_ref) * std::exp(-d2 / r2);
            }
}

}  // namespace scenarios

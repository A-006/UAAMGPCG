#include "scenarios/karman.h"
#include "bc/patches.h"
#include <cmath>

namespace scenarios {

void setup_karman_cylinder(Grid& g, const Karman& k) {
    double R2 = k.cyl_R * k.cyl_R;
    for (int j = 1; j <= g.ny; j++) {
        for (int i = 1; i <= g.nx; i++) {
            double xc = (i - 0.5) * g.dx;
            double yc = (j - 0.5) * g.dy;
            double rx = xc - k.cyl_cx;
            double ry = yc - k.cyl_cy;
            if (rx * rx + ry * ry < R2) g.set_solid(i, j);
        }
    }
}

void set_uniform_inflow(Grid& g, double U_inf) {
    for (int j = 1; j <= g.ny; j++)
        for (int i = 0; i <= g.nx; i++)
            g.u_at(i, j) = U_inf;
}

void seed_wake_perturbation(Grid& g, const Karman& k, double amplitude) {
    double eps = amplitude * k.U_inf;
    double x_lo = k.cyl_cx + k.cyl_R;
    double x_hi = k.cyl_cx + 5.0 * k.cyl_R;
    for (int j = 1; j <= g.ny; j++) {
        double y = (j - 0.5) * g.dy;
        for (int i = 1; i <= g.nx; i++) {
            double x = (i - 0.5) * g.dx;
            if (x > x_lo && x < x_hi)
                g.v_at(i, j - 1) += eps * std::sin(M_PI * (y - k.cyl_cy) / k.cyl_R);
        }
    }
}

bc::BoundaryManager karman_bcs(double U_inf) {
    // Reuses the BC factory defined in bc/patches.h — single place where
    // the inflow / outflow / free-slip / solid patches are wired together.
    return bc::karman(U_inf);
}

}  // namespace scenarios

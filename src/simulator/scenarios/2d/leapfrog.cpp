#include "simulator/scenarios/2d/leapfrog.h"
#include <cmath>

namespace scenarios {
namespace {

// Lamb-Oseen vortex velocity contribution at (x,y) from a vortex at (xc,yc).
void add_vortex(double x, double y, double xc, double yc, double gamma, double a, double& u,
                double& v) {
    double dx = x - xc, dy = y - yc;
    double r2 = dx * dx + dy * dy + 1e-12;
    double s  = gamma / (2.0 * M_PI) * (1.0 - std::exp(-r2 / (a * a))) / r2;
    u += -s * dy;
    v += s * dx;
}

} // namespace

void seed_vortex_dipoles(Grid& g, const VortexDipoles& d) {
    double yc = d.yc >= 0.0 ? d.yc : 0.5 * g.Ly();
    // Each dipole: top vortex −gamma, bottom vortex +gamma ⇒ self-propels +x.
    double xs[2] = {d.x0, d.x1};
    auto vel     = [&](double x, double y, double& u, double& v) {
        u = v = 0.0;
        for (double xc : xs) {
            add_vortex(x, y, xc, yc + d.half_d, -d.gamma, d.core, u, v); // top (clockwise)
            add_vortex(x, y, xc, yc - d.half_d, +d.gamma, d.core, u, v); // bottom (ccw)
        }
    };

    // u-faces at (i·dx, (j-0.5)·dy)
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            double u, v;
            vel(i * g.dx, (j - 0.5) * g.dy, u, v);
            g.u_at(i, j) += u;
        }
    // v-faces at ((i-0.5)·dx, j·dy)
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++) {
            double u, v;
            vel((i - 0.5) * g.dx, j * g.dy, u, v);
            g.v_at(i, j) += v;
        }
}

} // namespace scenarios

#include "simulator/scenarios/2d/leapfrog_scenario.h"
#include "numerics/bc/patches.h"
#include <cmath>

namespace scenarios {

void LeapfrogScenario::configure(Config& cfg) const {
    cfg.Lx              = 4.0;
    cfg.Ly              = 2.0;
    cfg.NY              = std::max(cfg.NX / 2, 16);
    cfg.dt              = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf; // CFL ~ 0.5
    cfg.Re              = 0.0;       // inviscid — pure impulse transport test
    cfg.time_integrator = "lfm";
    cfg.out_dir         = "output_leapfrog";
}

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

// Velocity of the full 4-vortex (two-dipole) configuration at a point.
void leapfrog_velocity(double x, double y, double Ly, double& u, double& v) {
    u = v = 0.0;
    double a  = 0.12;       // core radius
    double G  = 1.0;        // |circulation|
    double d  = 0.30;       // half-separation of each dipole (in y)
    double yc = 0.5 * Ly;   // axis of propagation
    // Leading dipole near x=1.4, trailing dipole near x=0.9.
    // Each dipole: top vortex −G, bottom vortex +G ⇒ self-propels toward +x.
    double xs[2] = {1.4, 0.9};
    for (double xc : xs) {
        add_vortex(x, y, xc, yc + d, -G, a, u, v); // top (clockwise)
        add_vortex(x, y, xc, yc - d, +G, a, u, v); // bottom (counter-clockwise)
    }
}
} // namespace

void LeapfrogScenario::init_grid(Grid& g, const Config&) const {
    double Ly = g.Ly();
    // u-faces at (i·dx, (j-0.5)·dy)
    for (int i = 0; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++) {
            double u, v;
            leapfrog_velocity(i * g.dx, (j - 0.5) * g.dy, Ly, u, v);
            g.u_at(i, j) = u;
        }
    // v-faces at ((i-0.5)·dx, j·dy)
    for (int i = 1; i <= g.nx; i++)
        for (int j = 0; j <= g.ny; j++) {
            double u, v;
            leapfrog_velocity((i - 0.5) * g.dx, j * g.dy, Ly, u, v);
            g.v_at(i, j) = v;
        }
}

bc::BoundaryManager LeapfrogScenario::boundary_manager(const Config&) const {
    return bc::free_slip_walls();
}

} // namespace scenarios

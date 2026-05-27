#include "scenarios/wind_turbine.h"
#include "scenarios/delta_wing.h"   // reuses InflowXMin3D, OutflowXMax3D, FreeSlipYZ3D
#include <cmath>

namespace scenarios {

namespace {

// Returns true iff point (xc,yc,zc) is inside any blade at rotor angle `angle`.
// Blades rotate about the x-axis through hub_center, lying in the y-z plane.
// Each blade is a thin rectangle of length blade_radius (radial) × chord
// (along x) × 2*thickness (along the tangential direction).
bool inside_blade(const WindTurbine& wt, double xc, double yc, double zc,
                  double angle, double& v_y, double& v_z) {
    double dy = yc - wt.hub_center[1];
    double dz = zc - wt.hub_center[2];
    double r  = std::sqrt(dy * dy + dz * dz);

    // Hub itself (small cylinder around the axis)
    if (r < wt.hub_radius
        && std::abs(xc - wt.hub_center[0]) < wt.chord * 0.5) {
        v_y = -wt.angular_velocity * dz;
        v_z =  wt.angular_velocity * dy;
        return true;
    }

    if (r < wt.hub_radius * 0.95) return false;
    if (r > wt.blade_radius)       return false;
    if (std::abs(xc - wt.hub_center[0]) > wt.chord * 0.5) return false;

    // Check each blade
    double phi = std::atan2(dz, dy);  // angle of point in y-z plane
    double dphi = 2.0 * M_PI / wt.n_blades;
    for (int b = 0; b < wt.n_blades; b++) {
        double blade_angle = angle + b * dphi;
        // Distance perpendicular to the blade axis, in the y-z plane
        double tangential = r * std::sin(phi - blade_angle);
        double radial     = r * std::cos(phi - blade_angle);
        if (radial > 0.0 && std::abs(tangential) < wt.thickness) {
            v_y = -wt.angular_velocity * dz;
            v_z =  wt.angular_velocity * dy;
            return true;
        }
    }
    return false;
}

}  // namespace

void apply_turbine_state(Grid3D& g, const WindTurbine& wt, double angle) {
    // Re-set solid mask + rotational velocity on the blade surfaces.
    // Clear previous solid first (turbine is the only solid in this scenario).
    std::fill(g.solid.begin(), g.solid.end(), false);

    for (int k = 1; k <= g.nz; k++) {
        double zc = (k - 0.5) * g.dz;
        for (int j = 1; j <= g.ny; j++) {
            double yc = (j - 0.5) * g.dy;
            for (int i = 1; i <= g.nx; i++) {
                double xc = (i - 0.5) * g.dx;
                double vy = 0, vz = 0;
                if (inside_blade(wt, xc, yc, zc, angle, vy, vz)) {
                    g.set_solid(i, j, k);
                    // Set the surrounding face velocities to the blade's
                    // local rigid-body rotation velocity (no-slip on the
                    // moving surface).
                    if (i > 1)        g.u_at(i - 1, j, k) = 0.0;
                    if (i < g.nx)     g.u_at(i,     j, k) = 0.0;
                    if (j > 1)        g.v_at(i, j - 1, k) = vy;
                    if (j < g.ny)     g.v_at(i, j,     k) = vy;
                    if (k > 1)        g.w_at(i, j, k - 1) = vz;
                    if (k < g.nz)     g.w_at(i, j, k)     = vz;
                }
            }
        }
    }
}

bc::BoundaryManager3D wind_turbine_bcs(double U_inf) {
    // Same patch set as the delta wing: inflow / outflow / free-slip walls
    // / no-slip immersed solid.
    return delta_wing_bcs(U_inf);
}

}  // namespace scenarios

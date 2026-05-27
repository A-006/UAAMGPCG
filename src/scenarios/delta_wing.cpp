#include "scenarios/delta_wing.h"
#include <cmath>

namespace scenarios {

namespace {

// 3D BC implementations specific to the wing scenario. These mirror the
// 2D Karman patches but for the 3D MAC layout.

class InflowXMin3D : public bc::BoundaryCondition3D {
public:
    explicit InflowXMin3D(double U_inf) : U_(U_inf) {}
    void apply(Grid3D& g) const override {
        for (int k = 1; k <= g.nz; k++)
            for (int j = 1; j <= g.ny; j++) {
                g.u_at(0, j, k) = U_;
                g.v_at(0, j, k) = 0.0;
                g.w_at(0, j, k) = 0.0;
            }
    }
    const char* name() const override { return "InflowXMin3D"; }
private:
    double U_;
};

class OutflowXMax3D : public bc::BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override {
        int nx = g.nx;
        for (int k = 1; k <= g.nz; k++)
            for (int j = 1; j <= g.ny; j++) {
                g.u_at(nx,     j, k) = g.u_at(nx - 1, j, k);
                g.v_at(nx + 1, j, k) = g.v_at(nx,     j, k);
                g.w_at(nx + 1, j, k) = g.w_at(nx,     j, k);
            }
    }
    const char* name() const override { return "OutflowXMax3D"; }
};

// Free-slip on the y=0/y=Ly and z=0/z=Lz walls.
class FreeSlipYZ3D : public bc::BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override {
        int nx = g.nx, ny = g.ny, nz = g.nz;
        // y faces
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
        // z faces
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
    const char* name() const override { return "FreeSlipYZ3D"; }
};

}  // namespace

void setup_delta_wing(Grid3D& g, const DeltaWing& wing) {
    double aoa = wing.aoa_deg * M_PI / 180.0;
    double cs = std::cos(aoa), sn = std::sin(aoa);
    double z_mid = 0.5 * g.Lz();

    for (int k = 1; k <= g.nz; k++) {
        double zc = (k - 0.5) * g.dz;
        for (int j = 1; j <= g.ny; j++) {
            double yc = (j - 0.5) * g.dy;
            for (int i = 1; i <= g.nx; i++) {
                double xc = (i - 0.5) * g.dx;

                // Translate to wing reference frame (apex at origin in x,
                // wing in z–chord plane at y = wing.y_mid).
                double dx = xc - wing.leading_x;
                double dy = yc - wing.y_mid;
                double dz = zc - z_mid;

                // Rotate about z-axis by -aoa: the wing is at angle of
                // attack, so the x-axis of the wing frame is the chord
                // direction rotated up by aoa in world coords.
                double xb =  dx * cs + dy * sn;
                double yb = -dx * sn + dy * cs;

                if (xb < 0.0 || xb > wing.chord) continue;
                // Triangular planform: at chord position xb, semi-span
                // tapers from 0 at apex to semi_span at root.
                double half_span_at_xb = wing.semi_span * (xb / wing.chord);
                if (std::abs(dz) > half_span_at_xb) continue;
                if (std::abs(yb) > wing.thickness)  continue;
                g.set_solid(i, j, k);
            }
        }
    }
}

void set_uniform_inflow(Grid3D& g, double U_inf) {
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++)
                g.u_at(i, j, k) = U_inf;
}

bc::BoundaryManager3D delta_wing_bcs(double U_inf) {
    bc::BoundaryManager3D mgr;
    mgr.add(std::make_unique<InflowXMin3D>(U_inf));
    mgr.add(std::make_unique<OutflowXMax3D>());
    mgr.add(std::make_unique<FreeSlipYZ3D>());
    mgr.add(std::make_unique<bc::NoSlipImmersedSolid3D>());
    return mgr;
}

}  // namespace scenarios

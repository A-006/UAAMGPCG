#include "io/3d/delta_wing.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace scenarios {

namespace {

// 3D BC implementations specific to the wing scenario. These mirror the
// 2D Cylinder patches but for the 3D MAC layout.

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
    const char* name() const override {
        return "InflowXMin3D";
    }

private:
    double U_;
};

class OutflowXMax3D : public bc::BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override {
        int nx = g.nx;
        for (int k = 1; k <= g.nz; k++)
            for (int j = 1; j <= g.ny; j++) {
                g.u_at(nx, j, k)     = g.u_at(nx - 1, j, k);
                g.v_at(nx + 1, j, k) = g.v_at(nx, j, k);
                g.w_at(nx + 1, j, k) = g.w_at(nx, j, k);
            }
    }
    const char* name() const override {
        return "OutflowXMax3D";
    }
};

// Free-slip on the y=0/y=Ly and z=0/z=Lz walls.
class FreeSlipYZ3D : public bc::BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override {
        int nx = g.nx, ny = g.ny, nz = g.nz;
        // y faces
        for (int k = 1; k <= nz; k++) {
            for (int i = 1; i <= nx; i++) {
                g.v_at(i, 0, k)  = 0.0;
                g.v_at(i, ny, k) = 0.0;
            }
            for (int i = 0; i <= nx; i++) {
                g.u_at(i, 0, k)      = g.u_at(i, 1, k);
                g.u_at(i, ny + 1, k) = g.u_at(i, ny, k);
            }
            for (int i = 1; i <= nx; i++) {
                g.w_at(i, 0, k)      = g.w_at(i, 1, k);
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
    const char* name() const override {
        return "FreeSlipYZ3D";
    }
};

// Prescribe a uniform freestream (Ux,Uy,Uz) on all six walls: the normal
// velocity on each wall face is the freestream normal component, and the
// tangential ghost layer is set to the freestream so tangential gradients
// vanish. Inflow flux = outflow flux exactly → mass-balanced (paper's BC).
class FreestreamBox3D : public bc::BoundaryCondition3D {
public:
    FreestreamBox3D(double ux, double uy, double uz) : Ux_(ux), Uy_(uy), Uz_(uz) {}
    void apply(Grid3D& g) const override {
        int nx = g.nx, ny = g.ny, nz = g.nz;
        // x = 0 and x = Lx walls
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++) {
                g.u_at(0, j, k)  = Ux_; // normal in
                g.u_at(nx, j, k) = Ux_; // normal out
                g.v_at(0, j, k)      = Uy_;
                g.v_at(nx + 1, j, k) = Uy_;
                g.w_at(0, j, k)      = Uz_;
                g.w_at(nx + 1, j, k) = Uz_;
            }
        // y = 0 and y = Ly walls
        for (int k = 1; k <= nz; k++)
            for (int i = 1; i <= nx; i++) {
                g.v_at(i, 0, k)  = Uy_;
                g.v_at(i, ny, k) = Uy_;
                g.u_at(i, 0, k)      = Ux_;
                g.u_at(i, ny + 1, k) = Ux_;
                g.w_at(i, 0, k)      = Uz_;
                g.w_at(i, ny + 1, k) = Uz_;
            }
        // z = 0 and z = Lz walls
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                g.w_at(i, j, 0)  = Uz_;
                g.w_at(i, j, nz) = Uz_;
                g.u_at(i, j, 0)      = Ux_;
                g.u_at(i, j, nz + 1) = Ux_;
                g.v_at(i, j, 0)      = Uy_;
                g.v_at(i, j, nz + 1) = Uy_;
            }
    }
    const char* name() const override {
        return "FreestreamBox3D";
    }

private:
    double Ux_, Uy_, Uz_;
};

} // namespace

void setup_delta_wing(Grid3D& g, const DeltaWing& wing) {
    double tilt = wing.tilt_deg * M_PI / 180.0;
    double cs = std::cos(tilt), sn = std::sin(tilt);
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
                double xb = dx * cs + dy * sn;
                double yb = -dx * sn + dy * cs;

                if (xb < 0.0 || xb > wing.chord)
                    continue;
                // Triangular planform: at chord position xb, semi-span
                // tapers from 0 at apex to semi_span at root.
                double half_span_at_xb = wing.semi_span * (xb / wing.chord);
                if (std::abs(dz) > half_span_at_xb)
                    continue;
                if (std::abs(yb) > wing.thickness)
                    continue;
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

void set_uniform_freestream(Grid3D& g, double Ux, double Uy, double Uz) {
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++)
                g.u_at(i, j, k) = Ux;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 0; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                g.v_at(i, j, k) = Uy;
    for (int k = 0; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                g.w_at(i, j, k) = Uz;
}

void load_sdf_solid(Grid3D& g, const std::string& npy_path) {
    std::ifstream f(npy_path, std::ios::binary);
    if (!f) {
        std::cerr << "load_sdf_solid: cannot open " << npy_path << "\n";
        return;
    }
    char magic[6];
    f.read(magic, 6); // \x93NUMPY
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] >= 2) {
        f.read(reinterpret_cast<char*>(&hlen), 4); // v2.0+: uint32 header length
    } else {
        uint16_t h16 = 0;
        f.read(reinterpret_cast<char*>(&h16), 2); // v1.0: uint16
        hlen = h16;
    }
    std::string header(hlen, ' ');
    f.read(&header[0], hlen); // we trust shape == grid dims (verified)

    long n = (long)g.nx * g.ny * g.nz;
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), n * (long)sizeof(float));

    // C-order (x,y,z): idx = x*(ny*nz) + y*nz + z. sdf < 0 → solid.
    long solid_cnt = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            for (int k = 1; k <= g.nz; k++) {
                long idx = (long)(i - 1) * g.ny * g.nz + (long)(j - 1) * g.nz + (k - 1);
                if (data[idx] < 0.0f) {
                    g.set_solid(i, j, k);
                    solid_cnt++;
                }
            }
    std::cout << "load_sdf_solid: " << solid_cnt << " solid cells from " << npy_path << "\n";
}

bc::BoundaryManager3D freestream_box_bcs(double Ux, double Uy, double Uz) {
    bc::BoundaryManager3D mgr;
    mgr.add(std::make_unique<FreestreamBox3D>(Ux, Uy, Uz));
    mgr.add(std::make_unique<bc::NoSlipImmersedSolid3D>());
    return mgr;
}

} // namespace scenarios

#include "io/3d/plate.h"
#include <cmath>

namespace scenarios {

void setup_plate(Grid3D& g, const Plate& plate) {
    double tilt = plate.tilt_deg * M_PI / 180.0;
    double cs = std::cos(tilt), sn = std::sin(tilt);
    double z_mid = 0.5 * g.Lz();

    for (int k = 1; k <= g.nz; k++) {
        double zc = (k - 0.5) * g.dz;
        for (int j = 1; j <= g.ny; j++) {
            double yc = (j - 0.5) * g.dy;
            for (int i = 1; i <= g.nx; i++) {
                double xc = (i - 0.5) * g.dx;

                // Translate to the plate reference frame (apex at origin in x,
                // plate in the z–chord plane at y = plate.y_mid).
                double dx = xc - plate.leading_x;
                double dy = yc - plate.y_mid;
                double dz = zc - z_mid;

                // Rotate about z by -tilt: the plate is tilted, so the chord
                // direction is rotated up by tilt in world coords.
                double xb = dx * cs + dy * sn;
                double yb = -dx * sn + dy * cs;

                if (xb < 0.0 || xb > plate.chord)
                    continue;
                // Triangular planform: semi-span tapers from 0 at the apex to
                // semi_span at the root.
                double half_span_at_xb = plate.semi_span * (xb / plate.chord);
                if (std::abs(dz) > half_span_at_xb)
                    continue;
                if (std::abs(yb) > plate.thickness)
                    continue;
                g.set_solid(i, j, k);
            }
        }
    }
}

} // namespace scenarios

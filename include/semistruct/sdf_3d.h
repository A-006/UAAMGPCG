// ─────────────────────────────────────────────────────────────────────────
// SDF helpers + refinement predicates for 3D adaptive grids (sphere narrow-band
// = the paper's sphere setup). RefineFn targets the finest level; grading fills
// the intermediate rings.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "adaptive_grid_3d.h"
#include <cmath>

namespace semistruct {

inline double sdfSphere(double x, double y, double z, double cx, double cy, double cz, double r) {
    double dx = x - cx, dy = y - cy, dz = z - cz;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - r;
}

inline AdaptiveGrid3D::RefineFn refineUniform3D() {
    return [](int, int, int, int, double, double, double, double) { return true; };
}

// Refine x < 0.5 to the finest level (single controlled T-junction wall).
inline AdaptiveGrid3D::RefineFn refineLeftHalf3D() {
    return [](int, int, int, int, double, double xc, double, double) { return xc < 0.5; };
}

// Narrow band around a sphere: refine cells with |sdf| < band (+h margin).
inline AdaptiveGrid3D::RefineFn refineNarrowBand3D(double cx, double cy, double cz,
                                                   double r, double band) {
    return [=](int, int, int, int, double h, double x, double y, double z) {
        return std::fabs(sdfSphere(x, y, z, cx, cy, cz, r)) < band + h;
    };
}

}  // namespace semistruct

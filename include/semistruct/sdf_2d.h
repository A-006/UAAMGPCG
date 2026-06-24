// ─────────────────────────────────────────────────────────────────────────
// Signed-distance helpers + refinement predicates for building adaptive grids.
//   * makeUniform     — every cell refined to the finest level (baseline)
//   * makeTwoLevel    — half the domain refined one level (clean single T-junction band)
//   * makeNarrowBand  — narrow-band refinement around an SDF iso-surface (the
//                        paper's sphere/star setup, 2D analogue: a circle)
// All predicates target the finest level L-1 and rely on AdaptiveGrid2D grading
// to insert the intermediate ring(s).
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "adaptive_grid_2d.h"
#include <cmath>

namespace semistruct {

inline double sdfCircle(double x, double y, double cx, double cy, double r) {
    return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - r;
}

// Refine everywhere → fully uniform finest grid.
inline AdaptiveGrid2D::RefineFn refineUniform() {
    return [](int, int, int, double, double, double) { return true; };
}

// Refine the left half (x < 0.5) to the finest level, leave the right coarse.
// Produces a single, well-controlled column of T-junctions.
inline AdaptiveGrid2D::RefineFn refineLeftHalf() {
    return [](int /*l*/, int /*i*/, int /*j*/, double h, double cxc, double /*cyc*/) {
        return cxc < 0.5;  // cell centre in left half
    };
}

// Narrow-band around a circle: refine a cell if it is within `band` of the
// surface |sdf| < band. Cells far from the surface stay coarse.
inline AdaptiveGrid2D::RefineFn refineNarrowBand(double cx, double cy, double r, double band) {
    return [=](int /*l*/, int /*i*/, int /*j*/, double h, double xc, double yc) {
        double phi = sdfCircle(xc, yc, cx, cy, r);
        return std::fabs(phi) < band + h;  // include cells whose footprint may cross the band
    };
}

}  // namespace semistruct

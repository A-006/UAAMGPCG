// Verification: discretization accuracy of the composite Laplacian operator
// (method of manufactured solutions). Reproduces the paper's Fig.7 study in 2D.
//
// f(x,y) = cos(2πx) cos(2πy),  -∇²f = 8π² cos(2πx) cos(2πy).
// This f has zero normal derivative on all four sides of [0,1]², so the natural
// Neumann FV boundary (no boundary flux) is EXACT there — isolating the interior
// + T-junction truncation error (matches the paper's Neumann-outer Fig.7 setup).
// We fit the volume-weighted RMS error vs cell size: ~2nd order on uniform grids,
// near-2nd order on adaptive grids.
#include "semistruct/adaptive_grid_2d.h"
#include "semistruct/poisson_operator_2d.h"
#include "semistruct/sdf_2d.h"
#include "../test_utils.h"
#include <cmath>
#include <cstdio>

using namespace semistruct;

static const double PI = 3.14159265358979323846;
static double fexact(double x, double y) { return std::cos(2 * PI * x) * std::cos(2 * PI * y); }
static double minusLap(double x, double y) {
    return 8 * PI * PI * std::cos(2 * PI * x) * std::cos(2 * PI * y);
}

// A DOF is "uniform-interior" if all four face neighbours are same-level leaves
// (no T-junction, no domain boundary touching it).
static bool isUniformInterior(const AdaptiveGrid2D& g, int d) {
    int l = g.dof_level[d], i = g.dof_i[d], j = g.dof_j[d];
    const int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
    for (int s = 0; s < 4; ++s) {
        int ni = i + di[s], nj = j + dj[s];
        if (!g.lev[l].in(ni, nj)) return false;
        if (g.lev[l].at(ni, nj) != Cell::LEAF) return false;
    }
    return true;
}

// Volume-weighted RMS of (numerical -∇²f) vs analytic, pure Neumann boundary.
// If `interior_only`, restrict to uniform-interior cells (away from T-junctions).
static double laplacianRMS(AdaptiveGrid2D& g, bool interior_only = false) {
    g.bc = {BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann};
    PoissonOperator2D op;
    op.build(g);
    std::vector<double> f = op.sample(fexact);
    std::vector<double> num = op.numericalLaplacian(f);  // (A f)/V ≈ -∇²f
    double sn = 0, sd = 0;
    for (int d = 0; d < g.ndof; ++d) {
        if (interior_only && !isUniformInterior(g, d)) continue;
        double x = g.cx(g.dof_level[d], g.dof_i[d]), y = g.cy(g.dof_level[d], g.dof_j[d]);
        double e = num[d] - minusLap(x, y);
        sn += op.vol[d] * e * e; sd += op.vol[d];
    }
    return std::sqrt(sn / sd);
}

static double fitRate(double e_coarse, double e_fine) { return std::log2(e_coarse / e_fine); }

int main() {
    test_header("Composite Laplacian accuracy (2D MMS)");

    // ── Uniform grids: 16,32,64,128 ──
    printf("  uniform:\n");
    double prev = 0;
    double rate_uniform = 0;
    for (int idx = 0; idx < 4; ++idx) {
        int n = 16 << idx;
        AdaptiveGrid2D g; g.build(1, n, refineUniform());
        double e = laplacianRMS(g);
        if (idx > 0) {
            rate_uniform = fitRate(prev, e);
            printf("    n=%-4d RMS=%.3e  rate=%.2f\n", n, e, rate_uniform);
        } else printf("    n=%-4d RMS=%.3e\n", n, e);
        prev = e;
    }
    check(rate_uniform > 1.8, "uniform Laplacian convergence ~2nd order");

    // ── Narrow-band adaptive grids (circle), increasing base resolution ──
    // Interior (away from T-junctions) is clean 2nd order. The all-cell RMS is
    // dominated by the O(n) T-junction cells, where the cell-centred FV Laplacian
    // has O(1) POINTWISE truncation (a known finite-volume property: the flux is
    // 2nd order but opposite-face fluxes have mismatched truncation). The
    // physically meaningful 2nd-order behaviour is on the SOLUTION
    // (test_poisson_convergence_2d), where the conservative scheme supraconverges.
    printf("  narrow-band (circle r=0.25):\n");
    double prev_int = 0, prev_all = 0, rate_int = 0, rate_all = 0;
    for (int idx = 0; idx < 3; ++idx) {
        int n0 = 16 << idx;
        AdaptiveGrid2D g; g.build(3, n0, refineNarrowBand(0.5, 0.5, 0.25, 0.08));
        double e_all = laplacianRMS(g, false);
        double e_int = laplacianRMS(g, true);
        if (idx > 0) { rate_int = fitRate(prev_int, e_int); rate_all = fitRate(prev_all, e_all); }
        printf("    n0=%-4d DOFs=%-7d  interior RMS=%.3e (rate=%.2f)  all RMS=%.3e (rate=%.2f)\n",
               n0, g.ndof, e_int, rate_int, e_all, rate_all);
        prev_int = e_int; prev_all = e_all;
    }
    check(rate_int > 1.8, "narrow-band interior Laplacian ~2nd order");
    check(rate_all > 0.3, "narrow-band all-cell Laplacian converges (T-junction O(1) pointwise)");

    return test_summary();
}

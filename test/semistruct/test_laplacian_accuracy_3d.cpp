// Verification: discretization accuracy of the composite Laplacian (3D MMS).
// f(x,y,z) = cos(2πx)cos(2πy)cos(2πz),  -∇²f = 12π² f  (zero normal derivative
// on all six faces ⇒ the natural Neumann FV boundary is exact). Interior cells
// (away from T-junctions) are ~2nd order; all-cell RMS is dominated by the O(n²)
// T-junction cells where the cell-centred FV Laplacian has O(1) pointwise
// truncation (a known FV property). The meaningful 2nd order is on the solution.
#include "semistruct/adaptive_grid_3d.h"
#include "semistruct/poisson_operator_3d.h"
#include "semistruct/sdf_3d.h"
#include "../test_utils.h"
#include <cmath>
#include <cstdio>

using namespace semistruct;
static const double PI = 3.14159265358979323846;
static double fexact(double x, double y, double z) {
    return std::cos(2*PI*x) * std::cos(2*PI*y) * std::cos(2*PI*z);
}
static double minusLap(double x, double y, double z) { return 12*PI*PI*fexact(x,y,z); }

static bool isUniformInterior(const AdaptiveGrid3D& g, int d) {
    int l = g.dof_level[d], i = g.dof_i[d], j = g.dof_j[d], k = g.dof_k[d];
    for (int s = 0; s < 6; ++s) {
        int ni = i+AdaptiveGrid3D::FDI[s], nj = j+AdaptiveGrid3D::FDJ[s], nk = k+AdaptiveGrid3D::FDK[s];
        if (!g.lev[l].in(ni, nj, nk)) return false;
        if (g.lev[l].at(ni, nj, nk) != Cell::LEAF) return false;
    }
    return true;
}

static double laplacianRMS(AdaptiveGrid3D& g, bool interior_only) {
    for (auto& b : g.bc) b = BC::Neumann;
    PoissonOperator3D op; op.build(g);
    std::vector<double> f = op.sample(fexact);
    std::vector<double> num = op.numericalLaplacian(f);
    double sn = 0, sd = 0;
    for (int d = 0; d < g.ndof; ++d) {
        if (interior_only && !isUniformInterior(g, d)) continue;
        double x = g.cx(g.dof_level[d], g.dof_i[d]), y = g.cy(g.dof_level[d], g.dof_j[d]),
               z = g.cz(g.dof_level[d], g.dof_k[d]);
        double e = num[d] - minusLap(x, y, z);
        sn += op.vol[d]*e*e; sd += op.vol[d];
    }
    return std::sqrt(sn / sd);
}
// Convergence rate from an error pair refined by `ratio` (res_fine/res_coarse).
static double rate(double ec, double ef, double ratio) { return std::log(ec / ef) / std::log(ratio); }

int main() {
    test_header("Composite Laplacian accuracy (3D MMS)");

    printf("  uniform:\n");
    double prev = 0, ru = 0; int prevn = 0;
    int ns[3] = {16, 24, 32};
    for (int idx = 0; idx < 3; ++idx) {
        AdaptiveGrid3D g; g.build(1, ns[idx], refineUniform3D());
        double e = laplacianRMS(g, false);
        if (idx > 0) { ru = rate(prev, e, (double)ns[idx]/prevn);
            printf("    n=%-3d RMS=%.3e rate=%.2f\n", ns[idx], e, ru); }
        else printf("    n=%-3d RMS=%.3e\n", ns[idx], e);
        prev = e; prevn = ns[idx];
    }
    check(ru > 1.8, "uniform Laplacian convergence ~2nd order");

    printf("  narrow-band sphere (r=0.25):\n");
    double pint = 0, pall = 0, rint = 0, rall = 0; int prevn0 = 0;
    int n0s[3] = {4, 6, 8};
    for (int idx = 0; idx < 3; ++idx) {
        AdaptiveGrid3D g; g.build(3, n0s[idx], refineNarrowBand3D(0.5,0.5,0.5,0.25,0.1));
        double ea = laplacianRMS(g, false), ei = laplacianRMS(g, true);
        double ratio = (double)n0s[idx] / prevn0;
        if (idx > 0) { rint = rate(pint, ei, ratio); rall = rate(pall, ea, ratio); }
        printf("    n0=%-3d DOFs=%-7d interior RMS=%.3e (rate=%.2f)  all RMS=%.3e (rate=%.2f)\n",
               n0s[idx], g.ndof, ei, rint, ea, rall);
        pint = ei; pall = ea; prevn0 = n0s[idx];
    }
    check(rint > 1.5, "narrow-band interior Laplacian near 2nd order");
    check(rall > 0.2, "narrow-band all-cell Laplacian converges (T-junction O(1) pointwise)");

    return test_summary();
}

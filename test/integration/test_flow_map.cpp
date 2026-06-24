// Flow maps (2D / 3D): identity initialization invariants.
//
// The forward map Φ and backward map Ψ both start as the identity ψ(x)=x with
// Jacobian = I. These tests verify set_identity / set_backward_identity place
// each cell-centered map at its own cell-center coordinate and set the Jacobian
// to the identity matrix.
//
// CPU-only. Links against liblfm_lib.a (no GPU/CUDA symbols).
#include "../test_utils.h"

#include "simulator/lfm/flow_map_2d.h"
#include "simulator/lfm/flow_map_3d.h"

#include <cmath>

int main() {
    test_header("Flow maps: identity initialization (2D + 3D)");

    const double EPS = 1e-12;

    // ===================================================================
    // 2D forward identity: Φ(X) = X at every cell center, F = I.
    // ===================================================================
    {
        FlowMap2D fm(7, 5, 1.5, 2.0); // dx=1.5, dy=2.0
        fm.set_identity();
        bool pos_ok = true, jac_ok = true;
        for (int j = 1; j <= fm.ny; ++j)
            for (int i = 1; i <= fm.nx; ++i) {
                size_t k = fm.idx(i, j);
                double x = (i - 0.5) * fm.dx;
                double y = (j - 0.5) * fm.dy;
                if (std::abs(fm.phi_x[k] - x) > EPS || std::abs(fm.phi_y[k] - y) > EPS)
                    pos_ok = false;
                if (std::abs(fm.F00[k] - 1.0) > EPS || std::abs(fm.F11[k] - 1.0) > EPS ||
                    std::abs(fm.F01[k]) > EPS || std::abs(fm.F10[k]) > EPS)
                    jac_ok = false;
            }
        check(pos_ok, "FlowMap2D::set_identity: Phi(X)=X at every cell center");
        check(jac_ok, "FlowMap2D::set_identity: forward Jacobian F = I");
    }

    // ===================================================================
    // 2D backward identity: Ψ(x) = x at every cell center, T = I.
    // ===================================================================
    {
        FlowMap2D fm(6, 8, 6.0, 8.0); // dx=dy=1.0
        fm.set_backward_identity();
        bool pos_ok = true, jac_ok = true;
        for (int j = 1; j <= fm.ny; ++j)
            for (int i = 1; i <= fm.nx; ++i) {
                size_t k = fm.idx(i, j);
                double x = (i - 0.5) * fm.dx;
                double y = (j - 0.5) * fm.dy;
                if (std::abs(fm.psi_x[k] - x) > EPS || std::abs(fm.psi_y[k] - y) > EPS)
                    pos_ok = false;
                if (std::abs(fm.T00[k] - 1.0) > EPS || std::abs(fm.T11[k] - 1.0) > EPS ||
                    std::abs(fm.T01[k]) > EPS || std::abs(fm.T10[k]) > EPS)
                    jac_ok = false;
            }
        check(pos_ok, "FlowMap2D::set_backward_identity: Psi(x)=x at every cell center");
        check(jac_ok, "FlowMap2D::set_backward_identity: backward Jacobian T = I");
    }

    // ===================================================================
    // 2D identity ⇒ pullback invariant: a field sampled at the (identity)
    // backward map coordinate equals the field at the original cell. Since
    // Ψ(x)=x, looking up a cell-centered field via psi returns input==output.
    // ===================================================================
    {
        FlowMap2D fm(5, 5, 5.0, 5.0);
        fm.set_backward_identity();
        // Cell-centered field f(x,y) = 3 + 2x - y (linear, arbitrary).
        auto field = [](double x, double y) { return 3.0 + 2.0 * x - y; };
        bool identical = true;
        for (int j = 1; j <= fm.ny; ++j)
            for (int i = 1; i <= fm.nx; ++i) {
                size_t k        = fm.idx(i, j);
                double orig     = field((i - 0.5) * fm.dx, (j - 0.5) * fm.dy);
                double pulled   = field(fm.psi_x[k], fm.psi_y[k]); // sample at Psi(x)=x
                if (std::abs(orig - pulled) > EPS)
                    identical = false;
            }
        check(identical, "FlowMap2D identity pullback: output == input field");
    }

    // ===================================================================
    // 3D forward identity: Φ(X) = X, F = I (full 3x3).
    // ===================================================================
    {
        FlowMap3D fm(4, 5, 3, 4.0, 10.0, 6.0); // dx=1, dy=2, dz=2
        fm.set_identity();
        bool pos_ok = true, diag_ok = true, off_ok = true;
        for (int k = 1; k <= fm.nz; ++k)
            for (int j = 1; j <= fm.ny; ++j)
                for (int i = 1; i <= fm.nx; ++i) {
                    size_t m = fm.idx(i, j, k);
                    double x = (i - 0.5) * fm.dx;
                    double y = (j - 0.5) * fm.dy;
                    double z = (k - 0.5) * fm.dz;
                    if (std::abs(fm.phi_x[m] - x) > EPS || std::abs(fm.phi_y[m] - y) > EPS ||
                        std::abs(fm.phi_z[m] - z) > EPS)
                        pos_ok = false;
                    if (std::abs(fm.F00[m] - 1.0) > EPS || std::abs(fm.F11[m] - 1.0) > EPS ||
                        std::abs(fm.F22[m] - 1.0) > EPS)
                        diag_ok = false;
                    if (std::abs(fm.F01[m]) > EPS || std::abs(fm.F02[m]) > EPS ||
                        std::abs(fm.F10[m]) > EPS || std::abs(fm.F12[m]) > EPS ||
                        std::abs(fm.F20[m]) > EPS || std::abs(fm.F21[m]) > EPS)
                        off_ok = false;
                }
        check(pos_ok, "FlowMap3D::set_identity: Phi(X)=X at every cell center");
        check(diag_ok, "FlowMap3D::set_identity: forward Jacobian diagonal = 1");
        check(off_ok, "FlowMap3D::set_identity: forward Jacobian off-diagonal = 0");
    }

    // ===================================================================
    // 3D backward identity: Ψ(x) = x, T = I (full 3x3).
    // ===================================================================
    {
        FlowMap3D fm(3, 3, 3, 3.0, 3.0, 3.0);
        fm.set_backward_identity();
        bool pos_ok = true, diag_ok = true, off_ok = true;
        for (int k = 1; k <= fm.nz; ++k)
            for (int j = 1; j <= fm.ny; ++j)
                for (int i = 1; i <= fm.nx; ++i) {
                    size_t m = fm.idx(i, j, k);
                    double x = (i - 0.5) * fm.dx;
                    double y = (j - 0.5) * fm.dy;
                    double z = (k - 0.5) * fm.dz;
                    if (std::abs(fm.psi_x[m] - x) > EPS || std::abs(fm.psi_y[m] - y) > EPS ||
                        std::abs(fm.psi_z[m] - z) > EPS)
                        pos_ok = false;
                    if (std::abs(fm.T00[m] - 1.0) > EPS || std::abs(fm.T11[m] - 1.0) > EPS ||
                        std::abs(fm.T22[m] - 1.0) > EPS)
                        diag_ok = false;
                    if (std::abs(fm.T01[m]) > EPS || std::abs(fm.T02[m]) > EPS ||
                        std::abs(fm.T10[m]) > EPS || std::abs(fm.T12[m]) > EPS ||
                        std::abs(fm.T20[m]) > EPS || std::abs(fm.T21[m]) > EPS)
                        off_ok = false;
                }
        check(pos_ok, "FlowMap3D::set_backward_identity: Psi(x)=x at every cell center");
        check(diag_ok, "FlowMap3D::set_backward_identity: backward Jacobian diagonal = 1");
        check(off_ok, "FlowMap3D::set_backward_identity: backward Jacobian off-diagonal = 0");
    }

    // ===================================================================
    // 3D identity ⇒ pullback invariant: field sampled at Psi(x)=x is unchanged.
    // ===================================================================
    {
        FlowMap3D fm(4, 4, 4, 4.0, 4.0, 4.0);
        fm.set_backward_identity();
        auto field = [](double x, double y, double z) { return 1.0 + x - 2.0 * y + 0.5 * z; };
        bool identical = true;
        for (int k = 1; k <= fm.nz; ++k)
            for (int j = 1; j <= fm.ny; ++j)
                for (int i = 1; i <= fm.nx; ++i) {
                    size_t m      = fm.idx(i, j, k);
                    double orig   = field((i - 0.5) * fm.dx, (j - 0.5) * fm.dy, (k - 0.5) * fm.dz);
                    double pulled = field(fm.psi_x[m], fm.psi_y[m], fm.psi_z[m]);
                    if (std::abs(orig - pulled) > EPS)
                        identical = false;
                }
        check(identical, "FlowMap3D identity pullback: output == input field");
    }

    // ===================================================================
    // idx() round-trip: distinct cells map to distinct flat indices, and the
    // index formula matches the documented column-major layout.
    // ===================================================================
    {
        FlowMap2D fm(7, 5, 7.0, 5.0);
        bool ok = true;
        for (int j = 1; j <= fm.ny; ++j)
            for (int i = 1; i <= fm.nx; ++i)
                if (fm.idx(i, j) != (size_t)((i - 1) + (j - 1) * fm.nx))
                    ok = false;
        check(ok, "FlowMap2D::idx matches column-major layout");

        FlowMap3D fm3(4, 5, 3, 4.0, 5.0, 3.0);
        bool ok3 = true;
        for (int k = 1; k <= fm3.nz; ++k)
            for (int j = 1; j <= fm3.ny; ++j)
                for (int i = 1; i <= fm3.nx; ++i)
                    if (fm3.idx(i, j, k) !=
                        (size_t)(i - 1) + (size_t)(j - 1) * fm3.nx +
                            (size_t)(k - 1) * fm3.nx * fm3.ny)
                        ok3 = false;
        check(ok3, "FlowMap3D::idx matches column-major layout");
    }

    return test_summary();
}

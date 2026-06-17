#pragma once
#include <cstddef>
#include <vector>

/// 2D flow map Φ and its Jacobian F = dΦ/dX, stored at cell centers.
/// Φ(X) maps initial position X to current position x.
/// F = [∂x/∂X, ∂x/∂Y; ∂y/∂X, ∂y/∂Y]
///
/// The cell-centered map (phi/F + psi/T) is used by the viscous path-integral
/// quadrature (Steps 4-17 of Algorithm 1). The impulse pullback (Steps 18-27)
/// uses the per-face FaceFlowMap2D below instead — a faithful 2D reduction of
/// the verified 3D FIX① design (author RKAxisKernel / PullbackAxisKernel),
/// which carries one covector per MAC face and writes the impulse straight onto
/// the velocity faces with NO cell-center↔face gauge averaging.
struct FlowMap2D {
    int nx, ny;
    double dx, dy;

    // Forward flow map: Φ(X) = x
    std::vector<double> phi_x; // x-coordinate after forward march
    std::vector<double> phi_y; // y-coordinate
    // Jacobian F = dΦ/dX (column-major: F00=dΦx/dX, F10=dΦy/dX, F01=dΦx/dY, F11=dΦy/dY)
    std::vector<double> F00, F10, F01, F11;

    // Backward flow map: Ψ(x) = X
    std::vector<double> psi_x;
    std::vector<double> psi_y;
    // Jacobian T = dΨ/dx
    std::vector<double> T00, T10, T01, T11;

    FlowMap2D(int nx_, int ny_, double dx_, double dy_);

    size_t idx(int i, int j) const {
        return (i - 1) + (j - 1) * nx;
    }

    void set_identity();
    void set_backward_identity();
};

/// Per-axis (staggered MAC) flow map — 2D reduction of the verified 3D
/// FaceFlowMap (FIX①). Each velocity face carries its own position and a
/// 2-component covector (= one column of the Jacobian). Sized to the u-face or
/// v-face grid by the owning simulator.
///   forward map:  position (fx, fy), covector (f0, f1)
///   backward map: position (bx, by), covector (t0, t1)
struct FaceFlowMap2D {
    std::vector<double> fx, fy, f0, f1; // forward (φ_a, F_a)
    std::vector<double> bx, by, t0, t1; // backward (ψ_a, T_a)
};

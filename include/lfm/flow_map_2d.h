#pragma once
#include <vector>
#include <cstddef>

/// 2D flow map Φ and its Jacobian F = dΦ/dX, stored at cell centers.
/// Φ(X) maps initial position X to current position x.
/// F = [∂x/∂X, ∂x/∂Y; ∂y/∂X, ∂y/∂Y]
struct FlowMap2D {
    int nx, ny;
    double dx, dy;

    // Forward flow map: Φ(X) = x
    std::vector<double> phi_x;   // x-coordinate after forward march
    std::vector<double> phi_y;   // y-coordinate
    // Jacobian F = dΦ/dX (column-major: F00=dΦx/dX, F10=dΦy/dX, F01=dΦx/dY, F11=dΦy/dY)
    std::vector<double> F00, F10, F01, F11;

    // Backward flow map: Ψ(x) = X
    std::vector<double> psi_x;
    std::vector<double> psi_y;
    // Jacobian T = dΨ/dx
    std::vector<double> T00, T10, T01, T11;

    FlowMap2D(int nx_, int ny_, double dx_, double dy_);

    size_t idx(int i, int j) const { return (i-1) + (j-1)*nx; }

    void set_identity();
    void set_backward_identity();
};

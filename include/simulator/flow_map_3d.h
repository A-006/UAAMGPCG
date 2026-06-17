#pragma once
#include <vector>
#include <cstddef>

/// 3D flow map Φ and its Jacobian F = dΦ/dX, stored at cell centers.
/// Φ(X) maps initial position X to current position x.
/// F is the 3×3 Jacobian with Fab = ∂Φ_a/∂X_b (a = row/output axis,
/// b = column/input axis; a,b ∈ {0,1,2} ≙ {x,y,z}). At identity F = I.
/// This mirrors include/simulator/flow_map_2d.h, doubled up for the extra axis.
struct FlowMap3D {
    int nx, ny, nz;
    double dx, dy, dz;

    // Forward flow map: Φ(X) = x
    std::vector<double> phi_x, phi_y, phi_z;
    // Jacobian F = dΦ/dX (9 comps, Fab = ∂Φ_a/∂X_b)
    std::vector<double> F00, F01, F02, F10, F11, F12, F20, F21, F22;

    // Backward flow map: Ψ(x) = X
    std::vector<double> psi_x, psi_y, psi_z;
    // Jacobian T = dΨ/dx (9 comps, Tab = ∂Ψ_a/∂x_b)
    std::vector<double> T00, T01, T02, T10, T11, T12, T20, T21, T22;

    FlowMap3D(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_);

    size_t idx(int i, int j, int k) const {
        return (size_t)(i - 1) + (size_t)(j - 1) * nx + (size_t)(k - 1) * nx * ny;
    }

    void set_identity();
    void set_backward_identity();
};

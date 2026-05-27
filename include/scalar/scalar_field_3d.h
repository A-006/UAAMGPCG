#pragma once
#include "core/grid_3d.h"
#include "advection/advection_3d.h"

// ──────────────────────────────────────────────────────────────────
// Cell-centered scalar field on a 3D MAC mesh (temperature, smoke, …).
//
// Pairs with Grid3D — the scalar lives on the same cells as the
// pressure field. Provides semi-Lagrangian advection (RK2 + trilinear)
// using the Grid3D's velocity, plus a Boussinesq-style buoyancy force
// that pushes mass upward proportional to scalar deviation from a base.
// ──────────────────────────────────────────────────────────────────
class ScalarField3D {
public:
    explicit ScalarField3D(const Grid3D& g) : mesh_(&g), data_(g.p_size(), 0.0) {}

    double& operator()(int i, int j, int k) { return data_[mesh_->ip(i, j, k)]; }
    double  operator()(int i, int j, int k) const { return data_[mesh_->ip(i, j, k)]; }

    std::vector<double>&       data()       { return data_; }
    const std::vector<double>& data() const { return data_; }

    void fill(double v) { for (auto& x : data_) x = v; }

    // RK2 semi-Lagrangian advection using the Grid3D velocity field.
    static void advect(const ScalarField3D& src, ScalarField3D& dst,
                       const Grid3D& flow, double dt);

    // Sample value at arbitrary physical position via trilinear interpolation.
    static double sample(const ScalarField3D& s, const Grid3D& mesh,
                         double x, double y, double z);

private:
    const Grid3D* mesh_;
    std::vector<double> data_;
};

// Apply Boussinesq buoyancy: w_face += dt * beta * (T_face − T_ref).
// Mixed onto w-faces by averaging the two adjacent cell-centered T values.
void apply_buoyancy(Grid3D& g, const ScalarField3D& T, double T_ref,
                    double beta, double dt);

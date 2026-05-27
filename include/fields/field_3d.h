#pragma once
#include "core/mesh_3d.h"
#include <vector>
#include <cstddef>

// ──────────────────────────────────────────────────────────────────
// Typed field on a 3D MAC mesh — parallels include/fields/field.h.
//
//   CellField3D   — cell-centered, (nx+2)(ny+2)(nz+2)
//   FaceXField3D  — u-face,        (nx+1)(ny+2)(nz+2)
//   FaceYField3D  — v-face,        (nx+2)(ny+1)(nz+2)
//   FaceZField3D  — w-face,        (nx+2)(ny+2)(nz+1)
// ──────────────────────────────────────────────────────────────────
namespace fields {

enum class Layout3D { Cell, FaceX, FaceY, FaceZ };

template <Layout3D L>
class Field3D {
public:
    explicit Field3D(const Mesh3D& mesh)
        : mesh_(&mesh), data_(layout_size(mesh)) {}

    double& operator()(int i, int j, int k) {
        return data_[flat_index(i, j, k)];
    }
    double operator()(int i, int j, int k) const {
        return data_[flat_index(i, j, k)];
    }

    std::vector<double>&       data()       { return data_; }
    const std::vector<double>& data() const { return data_; }
    const Mesh3D& mesh() const { return *mesh_; }
    std::size_t   size() const { return data_.size(); }

    void fill(double v) {
        for (auto& x : data_) x = v;
    }

private:
    const Mesh3D* mesh_;
    std::vector<double> data_;

    static int layout_size(const Mesh3D& m) {
        if constexpr (L == Layout3D::Cell)  return m.p_size();
        if constexpr (L == Layout3D::FaceX) return m.u_size();
        if constexpr (L == Layout3D::FaceY) return m.v_size();
        if constexpr (L == Layout3D::FaceZ) return m.w_size();
        return 0;
    }
    int flat_index(int i, int j, int k) const {
        if constexpr (L == Layout3D::Cell)  return mesh_->ip(i, j, k);
        if constexpr (L == Layout3D::FaceX) return mesh_->iu(i, j, k);
        if constexpr (L == Layout3D::FaceY) return mesh_->iv(i, j, k);
        if constexpr (L == Layout3D::FaceZ) return mesh_->iw(i, j, k);
        return 0;
    }
};

using CellField3D  = Field3D<Layout3D::Cell>;
using FaceXField3D = Field3D<Layout3D::FaceX>;
using FaceYField3D = Field3D<Layout3D::FaceY>;
using FaceZField3D = Field3D<Layout3D::FaceZ>;

}  // namespace fields

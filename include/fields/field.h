#pragma once
#include "core/mesh.h"
#include <vector>
#include <cstddef>

// ──────────────────────────────────────────────────────────────────
// Typed field on a 2D MAC mesh.
//
//   CellField   — values at cell centers, indexed by (i,j) ∈ [0..nx+1]×[0..ny+1]
//   FaceXField  — values on x-faces,      indexed by (i,j) ∈ [0..nx  ]×[0..ny+1]
//   FaceYField  — values on y-faces,      indexed by (i,j) ∈ [0..nx+1]×[0..ny  ]
//
// All fields hold a `double` for now; templating on the scalar type is
// straightforward once needed. The MAC layout is the same as the long-
// standing Grid::u, Grid::v, Grid::p vectors — Field is the typed wrapper
// that makes which-layout-is-this explicit at the type level.
//
// Grid (legacy facade) still owns its own std::vector data; Field is used
// by new code and operators where extra type safety helps.
// ──────────────────────────────────────────────────────────────────
namespace fields {

enum class Layout { Cell, FaceX, FaceY };

template <Layout L>
class Field {
public:
    explicit Field(const Mesh2D& mesh)
        : mesh_(&mesh), data_(layout_size(mesh)) {}

    // Element access with MAC-aware index
    double& operator()(int i, int j) {
        return data_[flat_index(i, j)];
    }
    double operator()(int i, int j) const {
        return data_[flat_index(i, j)];
    }

    // Raw vector view (for interop with legacy APIs that take vector<double>)
    std::vector<double>&       data()       { return data_; }
    const std::vector<double>& data() const { return data_; }

    const Mesh2D& mesh() const { return *mesh_; }
    std::size_t   size() const { return data_.size(); }

    void fill(double v) {
        for (auto& x : data_) x = v;
    }

private:
    const Mesh2D* mesh_;
    std::vector<double> data_;

    static int layout_size(const Mesh2D& m) {
        if constexpr (L == Layout::Cell)  return m.p_size();
        if constexpr (L == Layout::FaceX) return m.u_size();
        if constexpr (L == Layout::FaceY) return m.v_size();
        return 0;
    }
    int flat_index(int i, int j) const {
        if constexpr (L == Layout::Cell)  return mesh_->ip(i, j);
        if constexpr (L == Layout::FaceX) return mesh_->iu(i, j);
        if constexpr (L == Layout::FaceY) return mesh_->iv(i, j);
        return 0;
    }
};

using CellField  = Field<Layout::Cell>;
using FaceXField = Field<Layout::FaceX>;
using FaceYField = Field<Layout::FaceY>;

}  // namespace fields

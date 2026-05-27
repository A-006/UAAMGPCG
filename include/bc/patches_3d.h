#pragma once
#include "core/grid_3d.h"
#include <memory>
#include <vector>

// ──────────────────────────────────────────────────────────────────
// Patch-based BCs for 3D Grid3D — parallels include/bc/patches.h.
// ──────────────────────────────────────────────────────────────────
namespace bc {

class BoundaryCondition3D {
public:
    virtual ~BoundaryCondition3D() = default;
    virtual void apply(Grid3D& g) const = 0;
    virtual const char* name() const = 0;
};

class BoundaryManager3D {
public:
    BoundaryManager3D() = default;
    BoundaryManager3D(BoundaryManager3D&&) = default;
    BoundaryManager3D& operator=(BoundaryManager3D&&) = default;

    void add(std::unique_ptr<BoundaryCondition3D> bc) {
        if (bc) bcs_.push_back(std::move(bc));
    }
    void apply(Grid3D& g) const {
        for (const auto& bc : bcs_) bc->apply(g);
    }
    size_t size() const { return bcs_.size(); }
    bool empty() const { return bcs_.empty(); }

private:
    std::vector<std::unique_ptr<BoundaryCondition3D>> bcs_;
};

// Free-slip on all six faces — normal velocity = 0; tangential ∂/∂n = 0.
// Right BC for vortex-ring head-on collision, Taylor-Green decay, etc.
class FreeSlipAllFaces3D : public BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override;
    const char* name() const override { return "FreeSlipAllFaces3D"; }
};

// Periodic on all three axes — implements ghost-layer copy.
class Periodic3D : public BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override;
    const char* name() const override { return "Periodic3D"; }
};

// No-slip on every fluid-solid face inside the domain.
class NoSlipImmersedSolid3D : public BoundaryCondition3D {
public:
    void apply(Grid3D& g) const override;
    const char* name() const override { return "NoSlipImmersedSolid3D"; }
};

// ── Scenario builders ─────────────────────────────────────────────
BoundaryManager3D free_slip_box();  // free-slip 6 walls + immersed solid
BoundaryManager3D periodic_box();   // periodic on all 3 axes

}  // namespace bc

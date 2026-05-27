#pragma once
#include "core/grid.h"
#include <memory>
#include <vector>

// ──────────────────────────────────────────────────────────────────
// Patch-based boundary conditions (OpenFOAM-style).
//
// Each BoundaryCondition object owns one patch and knows how to enforce
// itself on a Grid. BoundaryManager composes a list of BCs; calling
// apply(g) walks them in order. This decouples scenario logic (which BCs
// to combine) from BC kernels.
// ──────────────────────────────────────────────────────────────────
namespace bc {

class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;
    virtual void apply(Grid& g) const = 0;
    virtual const char* name() const = 0;
};

// Composable list of BCs. Constructed once per simulator, applied every step.
class BoundaryManager {
public:
    BoundaryManager() = default;
    BoundaryManager(BoundaryManager&&) = default;
    BoundaryManager& operator=(BoundaryManager&&) = default;

    void add(std::unique_ptr<BoundaryCondition> bc) {
        if (bc) bcs_.push_back(std::move(bc));
    }
    void apply(Grid& g) const {
        for (const auto& bc : bcs_) bc->apply(g);
    }
    size_t size() const { return bcs_.size(); }
    bool empty() const { return bcs_.empty(); }

private:
    std::vector<std::unique_ptr<BoundaryCondition>> bcs_;
};

}  // namespace bc

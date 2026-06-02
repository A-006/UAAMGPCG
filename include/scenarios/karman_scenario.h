#pragma once
#include "scenarios/scenario.h"

namespace scenarios {

// 2D Kármán vortex street: immersed cylinder in a uniform stream
// (paper Fig. 8). Wraps the existing free-function building blocks in
// [karman.h]; carries no physics of its own.
class KarmanScenario : public Scenario {
public:
    void configure(Config& cfg) const override;
    void init_grid(Grid& g, const Config& cfg) const override;
    bc::BoundaryManager boundary_manager(const Config& cfg) const override;
    const char* name() const override {
        return "karman";
    }
};

} // namespace scenarios

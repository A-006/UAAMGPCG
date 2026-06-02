#pragma once
#include "simulator/scenarios/scenario.h"

namespace scenarios {

// Buoyant smoke in a closed box: no immersed solid, four no-slip walls,
// a constant upward body force on the fluid. Demonstrates the body-force
// hook and that adding a scenario touches nothing but its own files.
class SmokeScenario : public Scenario {
public:
    void configure(Config& cfg) const override;
    void init_grid(Grid& g, const Config& cfg) const override;
    bc::BoundaryManager boundary_manager(const Config& cfg) const override;
    void apply_body_force(Grid& g, const Config& cfg) const override;
    const char* name() const override {
        return "smoke";
    }
};

} // namespace scenarios

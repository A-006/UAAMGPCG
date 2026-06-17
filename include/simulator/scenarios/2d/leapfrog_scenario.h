#pragma once
#include "simulator/scenarios/scenario.h"

namespace scenarios {

// Leapfrogging vortex dipoles (paper Fig 10, 2D analogue of the leapfrog rings).
// Two co-axial counter-rotating vortex pairs in a closed free-slip box: each
// pair self-propels along +x; the trailing pair is drawn through the leading
// pair, they swap roles, and the motion repeats — the 2D "leapfrog". Inviscid
// (Re<=0), so the LFM impulse transport is what must keep the dipoles alive.
class LeapfrogScenario : public Scenario {
public:
    void configure(Config& cfg) const override;
    void init_grid(Grid& g, const Config& cfg) const override;
    bc::BoundaryManager boundary_manager(const Config& cfg) const override;
    const char* name() const override {
        return "leapfrog2d";
    }
};

} // namespace scenarios

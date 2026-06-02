#include "simulator/scenarios/scenario_3d.h"

namespace scenarios {

Scenario3DRegistry& Scenario3DRegistry::instance() {
    // No built-in 3D scenarios yet; a future 3D scenario registers here, e.g.
    //     r.add<VortexRingScenario3D>("vortex_ring");
    static Scenario3DRegistry registry;
    return registry;
}

} // namespace scenarios

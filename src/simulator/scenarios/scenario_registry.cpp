#include "simulator/scenarios/scenario_registry.h"
#include "simulator/scenarios/2d/karman_scenario.h"
#include "simulator/scenarios/2d/leapfrog_scenario.h"
#include "simulator/scenarios/2d/smoke_scenario.h"

namespace scenarios {

ScenarioRegistry& ScenarioRegistry::instance() {
    static ScenarioRegistry registry = [] {
        ScenarioRegistry r;
        r.add<KarmanScenario>("karman");
        r.add<LeapfrogScenario>("leapfrog2d");
        r.add<SmokeScenario>("smoke");
        return r;
    }();
    return registry;
}

} // namespace scenarios

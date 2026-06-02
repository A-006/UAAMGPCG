#include "scenarios/scenario_registry.h"
#include "scenarios/karman_scenario.h"
#include "scenarios/smoke_scenario.h"

namespace scenarios {

ScenarioRegistry& ScenarioRegistry::instance() {
    static ScenarioRegistry registry = [] {
        ScenarioRegistry r;
        r.add<KarmanScenario>("karman");
        r.add<SmokeScenario>("smoke");
        return r;
    }();
    return registry;
}

} // namespace scenarios

#include "simulator/scenarios/scenario_registry.h"
#include "simulator/scenarios/2d/generic_scenario.h"

namespace scenarios {

// 2D scenarios are data-driven: every scene is the same GenericScenario, with
// its presets/geometry/IC/BC coming from a case file (inputs/<name>.in). We
// register the built-in scene names so `scenario=<name>` validates and so the
// usage string lists them; the case file supplies all the behaviour. Any new
// scene is just a new inputs/<name>.in (build_config also accepts a name whose
// case file exists), so this list is for discoverability, not gatekeeping.
ScenarioRegistry& ScenarioRegistry::instance() {
    static ScenarioRegistry registry = [] {
        ScenarioRegistry r;
        r.add<GenericScenario>("karman");
        r.add<GenericScenario>("smoke");
        r.add<GenericScenario>("leapfrog2d");
        return r;
    }();
    return registry;
}

} // namespace scenarios

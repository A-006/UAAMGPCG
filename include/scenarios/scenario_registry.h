#pragma once
#include "scenarios/scenario.h"
#include "util/registry.h"

namespace scenarios {

// The process-wide registry of 2D scenarios. The singleton populates itself
// with the built-in scenarios (karman, smoke) on first access, so callers
// just do ScenarioRegistry::instance().create(name). Register additional
// scenarios with instance().add<MyScenario>("my-name").
class ScenarioRegistry : public util::Registry<Scenario> {
public:
    static ScenarioRegistry& instance();
};

} // namespace scenarios

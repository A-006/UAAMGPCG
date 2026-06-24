#pragma once
#include "io/ic/2d/generic_scenario.h"
#include "io/ic/scenario.h"
#include "util/registry.h"
#include <memory>

namespace scenarios {

// The process-wide registry of 2D scenarios. Every 2D scene is now the same
// data-driven GenericScenario (its presets/geometry/IC/BC come from a case file
// inputs/<name>.in), so `create()` always returns a GenericScenario regardless
// of name — the registered names exist only for `contains()`/`names()`
// (validation + usage listing). A new scene needs no registration: just add a
// case file. Callers do ScenarioRegistry::instance().create(name).
class ScenarioRegistry : public util::Registry<Scenario> {
public:
    static ScenarioRegistry& instance();

    // Data-driven: name selects the case file (loaded earlier into cfg), not the
    // class. Always a GenericScenario.
    std::unique_ptr<Scenario> create(const std::string&) const {
        return std::make_unique<GenericScenario>();
    }
};

} // namespace scenarios

#pragma once
#include "io/config.h"
#include <array>
#include <string>

// ──────────────────────────────────────────────────────────────────
// Index-style parameter reader for the data-driven 2D scenarios.
//
// A 2D case file describes its geometry / initial condition with a list of
// primitives, each addressed by a prefix and an index, e.g.
//
//     ic           = vortex_dipole       # the 0th IC source's KIND
//     ic0.gamma    = 1.0                 # the 0th source's params
//     ic1          = uniform_inflow      # a second source
//     ic1.U        = 1.0
//
// Param2D wraps cfg.extra so a primitive builder reads `gamma`, `U`, … under
// its own `<prefix><index>.` namespace, with the single-source shorthand
// `<prefix>.key` accepted when index 0 is the only source. Falls back to a
// caller-supplied default when a key is absent (so every primitive has sane
// built-in values and a case file only overrides what it needs).
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

class Param2D {
public:
    Param2D(const Config& cfg, std::string prefix, int index)
        : cfg_(cfg), prefix_(std::move(prefix)), index_(index) {}

    double d(const std::string& key, double def) const {
        return cfg_.dget(qualified(key), def);
    }
    int i(const std::string& key, int def) const {
        return cfg_.iget(qualified(key), def);
    }
    std::string s(const std::string& key, const std::string& def) const {
        return cfg_.sget(qualified(key), def);
    }
    std::array<double, 3> v3(const std::string& key, std::array<double, 3> def) const {
        return cfg_.v3get(qualified(key), def);
    }

private:
    // "<prefix><index>.<key>" if present, else the single-source "<prefix>.<key>".
    std::string qualified(const std::string& key) const {
        std::string indexed = prefix_ + std::to_string(index_) + "." + key;
        if (cfg_.extra.count(indexed))
            return indexed;
        return prefix_ + "." + key;
    }

    const Config& cfg_;
    std::string prefix_;
    int index_;
};

} // namespace scenarios

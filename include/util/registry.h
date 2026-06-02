#pragma once
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ──────────────────────────────────────────────────────────────────
// util::Registry<Base> — a tiny string-keyed factory for polymorphic
// objects. Shared by the scenario registries and the solver factories.
//
// Registration is explicit and ordinary: the owning singleton populates
// itself on first use, so there is no static-initialization-order
// fragility and nothing for the linker to elide.
// ──────────────────────────────────────────────────────────────────
namespace util {

template <class Base>
class Registry {
public:
    using Factory = std::function<std::unique_ptr<Base>()>;

    // Register a factory under `name`. Throws on a duplicate name.
    void add(const std::string& name, Factory factory) {
        if (!registry_.emplace(name, std::move(factory)).second)
            throw std::runtime_error("Registry: '" + name + "' already registered");
    }

    // Convenience overload: default-construct T on demand.
    template <class T>
    void add(const std::string& name) {
        add(name, [] { return std::make_unique<T>(); });
    }

    [[nodiscard]] bool contains(const std::string& name) const {
        return registry_.count(name) > 0;
    }

    // Build the object. Throws (listing known names) when `name` is absent.
    [[nodiscard]] std::unique_ptr<Base> create(const std::string& name) const {
        auto it = registry_.find(name);
        if (it == registry_.end())
            throw std::runtime_error("Registry: unknown '" + name + "'; known: " + joined_names());
        return it->second();
    }

    // Registered names, sorted — handy for usage strings and diagnostics.
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(registry_.size());
        for (const auto& [name, _] : registry_)
            out.push_back(name);
        std::sort(out.begin(), out.end());
        return out;
    }

private:
    std::string joined_names() const {
        std::string s;
        for (const auto& n : names())
            s += (s.empty() ? "" : ", ") + n;
        return s.empty() ? "(none registered)" : s;
    }

    std::unordered_map<std::string, Factory> registry_;
};

} // namespace util

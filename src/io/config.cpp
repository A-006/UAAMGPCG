#include "io/config.h"
// Most Config fields are a simple aggregate with inline defaults in the header.
// The `extra` map carries scenario-specific knobs read from INPUT files; the
// accessors below turn its string values into the type each scenario needs.
#include <cstddef>
#include <cstdlib>
#include <string>

namespace {

// Find a key in the extras map; return nullptr if absent.
const std::string* find(const std::map<std::string, std::string>& m, const std::string& key) {
    auto it = m.find(key);
    return it == m.end() ? nullptr : &it->second;
}

} // namespace

double Config::dget(const std::string& key, double def) const {
    const std::string* v = find(extra, key);
    if (!v || v->empty())
        return def;
    try {
        return std::stod(*v);
    } catch (...) {
        return def;
    }
}

int Config::iget(const std::string& key, int def) const {
    const std::string* v = find(extra, key);
    if (!v || v->empty())
        return def;
    try {
        return std::stoi(*v);
    } catch (...) {
        return def;
    }
}

std::string Config::sget(const std::string& key, const std::string& def) const {
    const std::string* v = find(extra, key);
    return (v && !v->empty()) ? *v : def;
}

std::array<double, 3> Config::v3get(const std::string& key, std::array<double, 3> def) const {
    const std::string* v = find(extra, key);
    if (!v || v->empty())
        return def;
    std::array<double, 3> out = def;
    std::size_t start         = 0;
    for (int i = 0; i < 3; i++) {
        std::size_t comma = v->find(',', start);
        std::string tok   = v->substr(start, comma - start);
        try {
            out[i] = std::stod(tok);
        } catch (...) {
            return def; // malformed triple → ignore the whole thing
        }
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return out;
}

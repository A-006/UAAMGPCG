#include "io/3d/case_3d.h"
#include "io/cli.h"
#include <climits>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace scene3d {

static void apply_presets(Config& cfg);
static std::string scenario_of(const config::KeyVals& kv);
static std::string find_preset(const std::string& scenario);

std::string peek_scenario(int argc, char** argv) {
    try {
        return scenario_of(config::collect_assignments(argc, argv));
    } catch (...) {
        return "vortex_ring"; // the chosen path reports the real error later
    }
}

std::string peek_key(int argc, char** argv, const std::string& key, const std::string& def) {
    try {
        std::string out = def;
        for (const auto& [k, v] : config::collect_assignments(argc, argv))
            if (k == key)
                out = v;
        return out;
    } catch (...) {
        return def; // the chosen path reports the real error later
    }
}

std::string peek_dim(int argc, char** argv) {
    try {
        std::string dim, scenario;
        bool have_dim = false;
        for (const auto& [k, v] : config::collect_assignments(argc, argv)) {
            if (k == "dim") {
                dim      = v;
                have_dim = true;
            } else if (k == "scenario")
                scenario = v;
        }
        if (have_dim)
            return dim; // explicit dim in file/CLI wins
        // Otherwise take dim from the scenario's case file (where dim=3 lives).
        if (!scenario.empty())
            if (std::string path = find_preset(scenario); !path.empty())
                for (const auto& [k, v] : config::read_assignments(path))
                    if (k == "dim")
                        dim = v;
        if (!dim.empty())
            return dim;
    } catch (...) {
    }
    return "2";
}

Config build_config(int argc, char** argv) {
    config::KeyVals kv = config::collect_assignments(argc, argv);

    // Lay down the scenario's presets, then apply the user's assignments on top
    // so any explicit override wins.
    Config cfg;
    cfg.scenario = scenario_of(kv);
    apply_presets(cfg);
    for (const auto& [k, v] : kv)
        config::set_field(cfg, k, v);
    return cfg;
}

// Directory holding the running executable (Linux /proc), or "" if unknown.
static std::string exe_dir() {
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        return "";
    buf[n]           = '\0';
    std::string path = buf;
    auto slash       = path.find_last_of('/');
    return slash == std::string::npos ? "" : path.substr(0, slash);
}

// Locate a scenario's preset file inputs/<scenario>.in. Search order:
// $UAAMG_PRESETS_DIR, the inputs/ dir beside (or one level above) the
// executable, then ./inputs. Returns "" if none of them has the file.
static std::string find_preset(const std::string& scenario) {
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("UAAMG_PRESETS_DIR"))
        dirs.push_back(env);
    if (std::string exe = exe_dir(); !exe.empty()) {
        dirs.push_back(exe + "/../inputs");
        dirs.push_back(exe + "/inputs");
    }
    dirs.push_back("inputs");

    for (const auto& dir : dirs) {
        std::string path = dir + "/" + scenario + ".in";
        if (std::ifstream(path).good())
            return path;
    }
    return "";
}

// Every 3D scenario shares the LFM base below; the rest of the recipe is DATA,
// read from its preset file inputs/<scenario>.in. build_config then layers the
// user's INI/CLI assignments on top so any explicit override still wins.
static void apply_presets(Config& cfg) {
    cfg.dim             = 3;
    cfg.time_integrator = "lfm";
    cfg.solver          = "cg"; // CPU backend: robust 3D choice (GPU has its own CG)
    cfg.extra["backend"] = "auto"; // GPU if available, else CPU (see main)

    std::string path = find_preset(cfg.scenario);
    if (path.empty())
        throw std::runtime_error("cfdsim: no preset for scenario '" + cfg.scenario +
                                 "'; expected inputs/" + cfg.scenario +
                                 ".in (or set UAAMG_PRESETS_DIR)");
    for (const auto& [key, value] : config::read_assignments(path))
        config::set_field(cfg, key, value);
}

// Which scenario do these assignments select? Last `scenario=` wins; the
// default applies when none is given.
static std::string scenario_of(const config::KeyVals& kv) {
    std::string scenario = "vortex_ring";
    for (const auto& [k, v] : kv)
        if (k == "scenario")
            scenario = v;
    return scenario;
}

} // namespace scene3d

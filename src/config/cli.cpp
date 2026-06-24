#include "config/cli.h"
#include "simulator/scenarios/2d/generic_scenario.h"
#include "simulator/scenarios/scenario_registry.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace config {
namespace {

using Assignment  = std::pair<std::string, std::string>;
using Assignments = std::vector<Assignment>;

// Stationary relaxation (jacobi/rbgs) needs many cheap sweeps to converge;
// Krylov / multigrid solvers need only a handful of expensive iterations.
int default_solve_iters(const std::string& solver) {
    return (solver == "jacobi" || solver == "rbgs") ? 2000 : 50;
}

std::string join(const std::vector<std::string>& xs) {
    std::string s;
    for (const auto& x : xs)
        s += (s.empty() ? "" : ", ") + x;
    return s;
}

std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// Read `key = value` lines (`#` starts a comment) into ordered assignments.
Assignments read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("config: cannot open file '" + path + "'");

    Assignments out;
    std::string line;
    for (int n = 1; std::getline(in, line); ++n) {
        if (auto hash = line.find('#'); hash != std::string::npos)
            line.erase(hash);
        if (trim(line).empty())
            continue;
        auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("config: " + path + ":" + std::to_string(n) +
                                     ": expected 'key = value'");
        out.emplace_back(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
    }
    return out;
}

std::string usage() {
    return "Usage: cfdsim [config.cfg] [key=value]...\n"
           "  e.g. cfdsim scenario=karman NX=128 t_end=2 solver=pcg time_integrator=lfm\n"
           "       cfdsim run.cfg NX=512        # file as base, CLI overrides\n"
           "  scenarios: " +
           join(scenarios::ScenarioRegistry::instance().names()) + "\n";
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// Last `scenario=` in the assignments wins; default if none.
std::string scenario_of(const Assignments& kv) {
    std::string s = "karman";
    for (const auto& [k, v] : kv)
        if (k == "scenario")
            s = v;
    return s;
}

// 2D scenes are data-driven: a bare `scenario=<name>` auto-loads inputs/<name>.in
// as the lowest-priority layer (explicit file/CLI assignments still win on top).
// Returns the case-file assignments, or empty if there is no matching file.
Assignments scenario_case_file(const Assignments& kv) {
    std::string path = "inputs/" + scenario_of(kv) + ".in";
    return file_exists(path) ? read_file(path) : Assignments{};
}

} // namespace

// Assemble a Config from an ordered list of assignments. The matching case file
// inputs/<scenario>.in seeds the presets/geometry/IC as the lowest layer; the
// GenericScenario then derives NY/dt; finally the explicit assignments are
// re-applied so anything the user wrote wins.
Config build_config(const KeyVals& explicit_kv) {
    // Layer order (low → high priority): scenario case file, then the caller's
    // explicit (file + CLI) assignments.
    KeyVals kv = scenario_case_file(explicit_kv);
    kv.insert(kv.end(), explicit_kv.begin(), explicit_kv.end());

    Config cfg;
    cfg.solver = "jacobi"; // default when unspecified

    auto apply_all = [&] {
        for (const auto& [key, value] : kv)
            set_field(cfg, key, value);
    };

    apply_all(); // pick up scenario / NX / U_inf / solver (inputs to the presets)

    // Valid if a registered built-in OR it has a case file inputs/<name>.in;
    // either way every 2D scene is the same data-driven GenericScenario.
    auto& registry = scenarios::ScenarioRegistry::instance();
    if (!registry.contains(cfg.scenario) && scenario_case_file(kv).empty())
        throw std::runtime_error("config: unknown scenario '" + cfg.scenario +
                                 "' (no inputs/" + cfg.scenario +
                                 ".in); known: " + join(registry.names()));
    scenarios::GenericScenario().configure(cfg);
    cfg.solve_iters = default_solve_iters(cfg.solver);

    apply_all(); // explicit values override the scenario-derived defaults
    return cfg;
}

Config load_file(const std::string& path) {
    return build_config(read_file(path));
}

Config build(const KeyVals& assignments) {
    return build_config(assignments);
}

KeyVals read_assignments(const std::string& path) {
    return read_file(path);
}

KeyVals collect_assignments(int argc, char* argv[]) {
    std::string file;
    KeyVals cli;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto eq         = arg.find('=');
        if (eq == std::string::npos) {
            if (!file.empty())
                throw std::runtime_error("config: more than one config file given ('" + file +
                                         "', '" + arg + "')");
            file = arg;
        } else {
            cli.emplace_back(arg.substr(0, eq), arg.substr(eq + 1));
        }
    }
    KeyVals all = file.empty() ? KeyVals{} : read_file(file);
    all.insert(all.end(), cli.begin(), cli.end()); // CLI after file ⇒ CLI wins
    return all;
}

Config build_config(int argc, char* argv[]) {
    return build_config(collect_assignments(argc, argv));
}

// One row per Config field: how to parse its string value. Adding a field is
// one line here — the same open/closed shape as the scenario registry. Any key
// NOT listed here is treated as a scenario-specific knob and stored in
// cfg.extra (read later via Config::dget / iget / sget).
void set_field(Config& cfg, const std::string& key, const std::string& value) {
    static const std::unordered_map<std::string, std::function<void(Config&, const std::string&)>>
        kSetters = {
            {"dim", [](Config& c, const std::string& v) { c.dim = std::stoi(v); }},
            {"NX", [](Config& c, const std::string& v) { c.NX = std::stoi(v); }},
            {"NY", [](Config& c, const std::string& v) { c.NY = std::stoi(v); }},
            {"NZ", [](Config& c, const std::string& v) { c.NZ = std::stoi(v); }},
            {"Lx", [](Config& c, const std::string& v) { c.Lx = std::stod(v); }},
            {"Ly", [](Config& c, const std::string& v) { c.Ly = std::stod(v); }},
            {"Lz", [](Config& c, const std::string& v) { c.Lz = std::stod(v); }},
            {"scenario", [](Config& c, const std::string& v) { c.scenario = v; }},
            {"U_inf", [](Config& c, const std::string& v) { c.U_inf = std::stod(v); }},
            {"Re", [](Config& c, const std::string& v) { c.Re = std::stod(v); }},
            {"cyl_cx", [](Config& c, const std::string& v) { c.cyl_cx = std::stod(v); }},
            {"cyl_cy", [](Config& c, const std::string& v) { c.cyl_cy = std::stod(v); }},
            {"cyl_cz", [](Config& c, const std::string& v) { c.cyl_cz = std::stod(v); }},
            {"cyl_R", [](Config& c, const std::string& v) { c.cyl_R = std::stod(v); }},
            {"cylinder_type", [](Config& c, const std::string& v) { c.cylinder_type = v; }},
            {"time_integrator", [](Config& c, const std::string& v) { c.time_integrator = v; }},
            {"dt", [](Config& c, const std::string& v) { c.dt = std::stod(v); }},
            {"t_end", [](Config& c, const std::string& v) { c.t_end = std::stod(v); }},
            {"lfm_cycle_steps",
             [](Config& c, const std::string& v) { c.lfm_cycle_steps = std::stoi(v); }},
            {"lfm_bfecc_clamp",
             [](Config& c, const std::string& v) { c.lfm_bfecc_clamp = std::stoi(v) != 0; }},
            {"lfm_march_fp32",
             [](Config& c, const std::string& v) { c.lfm_march_fp32 = std::stoi(v) != 0; }},
            {"lfm_bc", [](Config& c, const std::string& v) { c.lfm_bc = v; }},
            {"inflow_ux", [](Config& c, const std::string& v) { c.inflow_ux = std::stod(v); }},
            {"inflow_uy", [](Config& c, const std::string& v) { c.inflow_uy = std::stod(v); }},
            {"inflow_uz", [](Config& c, const std::string& v) { c.inflow_uz = std::stod(v); }},
            {"solver", [](Config& c, const std::string& v) { c.solver = v; }},
            {"solve_iters", [](Config& c, const std::string& v) { c.solve_iters = std::stoi(v); }},
            {"solve_tol", [](Config& c, const std::string& v) { c.solve_tol = std::stod(v); }},
            {"frame_skip", [](Config& c, const std::string& v) { c.frame_skip = std::stoi(v); }},
            {"out_dir", [](Config& c, const std::string& v) { c.out_dir = v; }},
        };

    auto it = kSetters.find(key);
    if (it == kSetters.end()) {
        // Not a core field → scenario-specific knob (see Config::extra). The 3D
        // scenarios read these via cfg.dget/iget/sget.
        cfg.extra[key] = value;
        return;
    }
    try {
        it->second(cfg, value);
    } catch (const std::logic_error&) { // stoi/stod: invalid_argument / out_of_range
        throw std::runtime_error("config: bad value '" + value + "' for key '" + key + "'");
    }
}

std::optional<Config> parse_cli(int argc, char* argv[]) {
    try {
        return build_config(collect_assignments(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n" << usage();
        return std::nullopt;
    }
}

} // namespace config

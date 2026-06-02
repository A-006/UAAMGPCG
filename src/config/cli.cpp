#include "config/cli.h"
#include "scenarios/scenario_registry.h"
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

// One row per Config field: how to parse its string value. Adding a field is
// one line here — the same open/closed shape as the scenario registry.
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
            {"solver", [](Config& c, const std::string& v) { c.solver = v; }},
            {"solve_iters", [](Config& c, const std::string& v) { c.solve_iters = std::stoi(v); }},
            {"solve_tol", [](Config& c, const std::string& v) { c.solve_tol = std::stod(v); }},
            {"frame_skip", [](Config& c, const std::string& v) { c.frame_skip = std::stoi(v); }},
            {"out_dir", [](Config& c, const std::string& v) { c.out_dir = v; }},
        };

    auto it = kSetters.find(key);
    if (it == kSetters.end())
        throw std::runtime_error("config: unknown key '" + key + "'");
    try {
        it->second(cfg, value);
    } catch (const std::logic_error&) { // stoi/stod: invalid_argument / out_of_range
        throw std::runtime_error("config: bad value '" + value + "' for key '" + key + "'");
    }
}

std::string join(const std::vector<std::string>& xs) {
    std::string s;
    for (const auto& x : xs)
        s += (s.empty() ? "" : ", ") + x;
    return s;
}

// Assemble a Config from an ordered list of assignments. Scenario presets fill
// the derived fields (Ly/NY/dt/out_dir/solve_iters) as defaults; the explicit
// assignments are then re-applied so anything the user wrote wins.
Config build_config(const Assignments& kv) {
    Config cfg;
    cfg.solver = "jacobi"; // default when unspecified

    auto apply_all = [&] {
        for (const auto& [key, value] : kv)
            set_field(cfg, key, value);
    };

    apply_all(); // pick up scenario / NX / U_inf / solver (inputs to the presets)

    auto& registry = scenarios::ScenarioRegistry::instance();
    if (!registry.contains(cfg.scenario))
        throw std::runtime_error("config: unknown scenario '" + cfg.scenario +
                                 "'; known: " + join(registry.names()));
    registry.create(cfg.scenario)->configure(cfg);
    cfg.solve_iters = default_solve_iters(cfg.solver);

    apply_all(); // explicit values override the scenario-derived defaults
    return cfg;
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
    return "Usage: lfm_2d [config.cfg] [key=value]...\n"
           "  e.g. lfm_2d scenario=karman NX=128 t_end=2 solver=pcg time_integrator=lfm\n"
           "       lfm_2d run.cfg NX=512        # file as base, CLI overrides\n"
           "  scenarios: " +
           join(scenarios::ScenarioRegistry::instance().names()) + "\n";
}

} // namespace

Config load_file(const std::string& path) {
    return build_config(read_file(path));
}

std::optional<Config> parse_cli(int argc, char* argv[]) {
    try {
        std::string file;
        Assignments cli;
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

        Assignments all = file.empty() ? Assignments{} : read_file(file);
        all.insert(all.end(), cli.begin(), cli.end()); // CLI after file ⇒ CLI wins
        return build_config(all);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n" << usage();
        return std::nullopt;
    }
}

} // namespace config

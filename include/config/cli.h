#pragma once
#include "config/config.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace config {

// Ordered (key, value) assignments parsed from an INI file and/or `key=value`
// command-line arguments. Later entries override earlier ones.
using KeyVals = std::vector<std::pair<std::string, std::string>>;

// Build a Config from the command line. The first non-`key=value` argument is
// an optional config-file path (INI-style `key = value`, `#` comments); any
// `key=value` arguments override fields from the file (and the file overrides
// scenario presets). Returns std::nullopt (after printing usage to stderr) on
// an unknown scenario / key / unreadable file.
//
//   lfm_2d [config.cfg] [key=value]...
//   lfm_2d scenario=karman NX=128 solver=pcg time_integrator=lfm
std::optional<Config> parse_cli(int argc, char* argv[]);

// Load a Config from an INI-style file (`key = value` per line, `#` comments).
// Fields left unset fall back to the scenario presets / Config defaults.
// Throws std::runtime_error on an unreadable file, unknown key, or bad value.
Config load_file(const std::string& path);

// ── Building blocks for callers that need their own preset logic ──────────
// The 2D `lfm_2d` path above runs the scenario through the 2D ScenarioRegistry.
// The 3D GPU launcher (`cfdsim`) instead applies its own per-scenario presets,
// so it consumes these two helpers directly rather than going through the
// registry-validating build_config().

// Collect assignments from argv: an optional INI-file path (the first argument
// without an `=`) followed by any number of `key=value` overrides (CLI wins
// over the file). Throws std::runtime_error on an unreadable file or a second
// file argument.
KeyVals collect_assignments(int argc, char* argv[]);

// Assemble a Config from an ordered list of assignments (the throwing core of
// parse_cli): apply the scenario's 2D presets, then the assignments on top so
// any user-set field wins. Throws std::runtime_error on an unknown scenario /
// key / bad value. Pair with collect_assignments to drive it from argv.
Config build_config(const KeyVals& kv);

// Same, straight from argv (collect_assignments + build_config in one call), so
// the launcher can mirror scene3d::build_config(argc, argv) for the 2D path.
// Throws on an unreadable file / unknown scenario / key / bad value.
Config build_config(int argc, char* argv[]);

// Apply one `key = value` onto cfg. Keys matching a core Config field are set
// directly (and type-checked); any other key is stored in cfg.extra, to be read
// later through Config::dget / iget / sget. Throws std::runtime_error on a bad
// value for a known key.
void set_field(Config& cfg, const std::string& key, const std::string& value);

} // namespace config

#pragma once
#include "config/config.h"
#include <optional>
#include <string>

namespace config {

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

} // namespace config

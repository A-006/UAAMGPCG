#pragma once
#include "config/config.h"
#include <optional>

namespace config {

// Parse command-line arguments and apply the per-scenario presets, producing a
// ready-to-use Config. Returns std::nullopt (after printing usage to stderr)
// when the scenario name is unknown.
//
//   lfm_2d [karman|smoke] [NX] [t_end] [solver]
std::optional<Config> parse_cli(int argc, char* argv[]);

} // namespace config

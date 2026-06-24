#pragma once
#include "io/config.h"
#include <string>

// ── 3D case-config assembly for the launcher ────────────────────────────────
// Turn argv (INI file + key=value) into a ready-to-run 3D Config: apply the LFM
// base defaults, load the case file inputs/<scenario>.in, then the user's
// overrides. Plus the peek helpers the launcher uses to pick the 2D-vs-3D path
// before full Config validation.
namespace scene3d {

// Peek the scenario name from argv without throwing (default "vortex_ring").
std::string peek_scenario(int argc, char** argv);

// Peek any key's value from argv (default `def` if absent / unparseable).
std::string peek_key(int argc, char** argv, const std::string& key, const std::string& def);

// Resolve dimensionality ("2"/"3") for the 2D-vs-3D dispatch: explicit `dim`
// wins, else read it from the scenario's case file; default "2".
std::string peek_dim(int argc, char** argv);

// Assemble the full 3D Config from argv. Throws on an unknown scenario / key.
Config build_config(int argc, char** argv);

} // namespace scene3d

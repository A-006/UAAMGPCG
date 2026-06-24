#pragma once
#include "core/config.h"
#include "io/cli.h"

// ── 2D case-config assembly for the launcher ────────────────────────────────
// The peer of io/3d/case_3d.h: turn argv (INI file + key=value) into a 2D
// Config. The 2D simulator itself is provisioned by make_simulator_2d.cpp.
namespace scene2d {

// Assemble a Config from argv (config file + key=value overrides, CLI wins).
inline Config build_config(int argc, char** argv) {
    return config::build_config(argc, argv);
}

} // namespace scene2d

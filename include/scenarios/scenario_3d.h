#pragma once
#include "bc/patches_3d.h"
#include "config/config.h"
#include "core/grid_3d.h"
#include "util/registry.h"

// ──────────────────────────────────────────────────────────────────
// 3D scenario provisioning — parallels the 2D Scenario interface
// ([scenario.h]) but is typed on Grid3D / bc::BoundaryManager3D, matching
// the codebase's 2D/3D duplication convention (Grid/Grid3D,
// Simulator/Simulator3D, Factory/Factory3D) rather than templating across
// dimensions.
//
// The interface and registry exist so a future 3D simulator can consume
// scenarios exactly as the 2D side does. No 3D scenarios are registered
// yet, so Scenario3DRegistry::instance().create(...) throws. Today's 3D
// tools still build their flow directly via the free functions in
// scenarios/*.h plus ChorinSimulator3D::mutable_grid() /
// set_boundary_manager() — see tools/run_vortex_ring_collision.cpp. A
// future 3D scenario subclasses Scenario3D and registers in instance().
// ──────────────────────────────────────────────────────────────────
namespace scenarios {

class Scenario3D {
public:
    virtual ~Scenario3D() = default;

    virtual void configure(Config& cfg) const                               = 0;
    virtual void init_grid(Grid3D& g, const Config& cfg) const              = 0;
    virtual bc::BoundaryManager3D boundary_manager(const Config& cfg) const = 0;
    virtual void apply_body_force(Grid3D&, const Config&) const {}
    virtual const char* name() const = 0;
};

// Process-wide registry of 3D scenarios. Currently empty (see header note).
class Scenario3DRegistry : public util::Registry<Scenario3D> {
public:
    static Scenario3DRegistry& instance();
};

} // namespace scenarios

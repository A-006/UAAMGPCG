#pragma once
#include "io/config.h"
#include "mesh/grid_3d.h"
#include "integrator/simulator_3d.h"
#include "solver/cuda/cuda_lfm_3d.h"

// ════════════════════════════════════════════════════════════════════
// Fully GPU-resident 3D LFM simulator (Simulator3D). The entire
// reinitialization cycle (advection, flow-map marching, viscous, pullback,
// gauge, projection) runs on the device; only the velocity field is copied
// back to a host Grid3D mirror once per cycle (for VTK / diagnostics).
//
// Usage mirrors LFMSimulator3D but BCs are the GPU free-slip box:
//   CudaLFMSimulator3D sim(cfg);
//   add_vortex_ring(sim.mutable_grid(), ring);  // set IC on the host mirror
//   sim.commit();                               // upload IC + apply BC
//   sim.step();                                 // one GPU LFM cycle
//
// Only includable from CUDA (.cu) translation units (the header pulls in
// CUDA types), matching the CudaPCGSolver3D convention.
// ════════════════════════════════════════════════════════════════════
class CudaLFMSimulator3D : public Simulator3D {
public:
    explicit CudaLFMSimulator3D(const Config& cfg);
    ~CudaLFMSimulator3D() override;

    void step() override;
    const Grid3D& grid() const override {
        return grid_;
    }
    double time() const override {
        return t_;
    }
    int step_count() const override {
        return step_;
    }

    Grid3D& mutable_grid() {
        return grid_;
    }
    // Upload the host grid's velocity + solid mask to the device and apply the
    // free-slip box BC. Call once after setting the initial condition.
    void commit();

    void run_cycle(int n_steps);

    // Test/diagnostic access to the device-resident state.
    CudaLFMState3D& state() {
        return s_;
    }

private:
    Config cfg_;
    Grid3D grid_;     // host mirror (IC in, fields out)
    CudaLFMState3D s_; // device-resident state
    double t_ = 0;
    int step_ = 0;
    void sync_to_host();
    void apply_bc(CudaVel3D v); // dispatch free_slip vs freestream wall BC
};

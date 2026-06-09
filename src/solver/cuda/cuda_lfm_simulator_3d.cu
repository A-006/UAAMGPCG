// ════════════════════════════════════════════════════════════════════
// Host orchestration of the fully GPU-resident 3D LFM cycle. Mirrors
// LFMSimulator3D::run_cycle (src/simulator/lfm_simulator_3d.cpp) line-for-line,
// but every sub-step is a kernel launch / device Poisson solve — the field
// never leaves the GPU until sync_to_host().
// ════════════════════════════════════════════════════════════════════
#include "simulator/cuda_lfm_simulator_3d.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>

// ── Per-operation LFM profiler (hotspot breakdown) ──
// Gated by env LFM_PROFILE: when set, each wrapped op syncs the device and
// accumulates wall time + call count; the table is printed at program exit.
// When unset, P(...) calls the op directly with zero overhead.
namespace {
struct LfmProf {
    struct E { double ms = 0; long n = 0; };
    std::map<std::string, E> e;
    bool on = (std::getenv("LFM_PROFILE") != nullptr);
    void rec(const char* k, double ms) {
        auto& x = e[k]; x.ms += ms; x.n++;
    }
    ~LfmProf() {
        if (!on || e.empty()) return;
        double tot = 0; for (auto& kv : e) tot += kv.second.ms;
        std::printf("\n=== LFM per-op profile (device-synced; abs time is sync-inflated, use %% / ms-call) ===\n");
        std::printf("  %-16s %10s %8s %12s %7s\n", "op", "total(ms)", "calls", "ms/call", "%");
        for (auto& kv : e) {
            auto& x = kv.second;
            std::printf("  %-16s %10.1f %8ld %12.4f %6.1f%%\n", kv.first.c_str(), x.ms, x.n,
                        x.ms / x.n, 100.0 * x.ms / tot);
        }
        std::printf("  %-16s %10.1f ms total\n", "TOTAL", tot);
    }
};
LfmProf g_lfm_prof;
template <class F> inline void prof(const char* k, F&& f) {
    if (!g_lfm_prof.on) { f(); return; }
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    g_lfm_prof.rec(k, std::chrono::duration<double, std::milli>(t1 - t0).count());
}
} // namespace
#define P(name, ...) prof(name, [&]() { __VA_ARGS__; })

CudaLFMSimulator3D::CudaLFMSimulator3D(const Config& cfg)
    : cfg_(cfg), grid_(cfg.NX, cfg.NY, cfg.NZ, cfg.Lx, cfg.Ly, cfg.Lz) {
    int n_steps = std::max(1, cfg.lfm_cycle_steps);
    s_.allocate(cfg.NX, cfg.NY, cfg.NZ, cfg.Lx / cfg.NX, cfg.Ly / cfg.NY, cfg.Lz / cfg.NZ, n_steps);
}

CudaLFMSimulator3D::~CudaLFMSimulator3D() {
    s_.free();
}

void CudaLFMSimulator3D::commit() {
    std::vector<char> solid(grid_.solid.size());
    for (size_t i = 0; i < solid.size(); i++)
        solid[i] = grid_.solid[i] ? 1 : 0;
    s_.upload_solid(solid);
    s_.upload_velocity(grid_.u, grid_.v, grid_.w);
    apply_bc(s_.cur);
    sync_to_host();
}

void CudaLFMSimulator3D::sync_to_host() {
    s_.download_velocity(grid_.u, grid_.v, grid_.w);
}

// Velocity wall BC dispatch: closed free-slip box (default) or prescribed
// freestream on all walls (delta wing / wind tunnel). Both also enforce
// no-slip on the immersed solid.
void CudaLFMSimulator3D::apply_bc(CudaVel3D v) {
    if (cfg_.lfm_bc == "freestream")
        lfm_apply_freestream_box(s_, v, cfg_.inflow_ux, cfg_.inflow_uy, cfg_.inflow_uz);
    else
        lfm_apply_free_slip_box(s_, v);
}

void CudaLFMSimulator3D::step() {
    run_cycle(cfg_.lfm_cycle_steps);
}

// ── Algorithm 1 (3D), GPU edition ──
void CudaLFMSimulator3D::run_cycle(int n_steps) {
    const double dt  = cfg_.dt;
    const double rho = 1.0;
    const double mu  = (cfg_.Re > 0) ? cfg_.U_inf * 2.0 * cfg_.cyl_R / cfg_.Re : 0.0;
    const int iters  = cfg_.solve_iters;
    const double tol = cfg_.solve_tol;

    CudaVel3D cur = s_.cur, u0 = s_.u0, A = s_.A, B = s_.B, C = s_.C;

    // u0_grid = grid_; optional half-step viscous (Step 1).
    P("copy", lfm_copy_vel(s_, u0, cur));
    if (mu > 0) {
        P("viscous", lfm_compute_viscous(s_, u0, mu));
        P("accum", lfm_accumulate_to_u0(s_, u0, s_.visc_x, s_.visc_y, s_.visc_z, dt / (2.0 * rho)));
    }

    // Cell-centered forward map (phi/F) + midpoints feed ONLY the viscous path
    // integral (lfm_accumulate_path_integral, mu>0). The FIX① per-face maps below
    // are independent. So for inviscid runs (collision) this whole chain is dead
    // work — skip it (saves ~12% of the cycle; the impulse pullback is unaffected).
    if (mu > 0)
        P("set_identity", lfm_set_identity(s_));

    // ── Steps 2-5: first midpoint u_{1/2} (= A) ──
    P("copy", lfm_copy_vel(s_, A, cur));
    P("advect", lfm_rk2_advect(s_, A, cur, cur, dt / 2.0));
    P("bc", apply_bc(A));
    P("project", lfm_project(s_, A, dt, iters, tol));
    P("bc", apply_bc(A));
    P("copy", lfm_copy_vel(s_, s_.vb[0], A));
    if (mu > 0) {
        P("save_fm", lfm_save_flow_map_state(s_));
        P("march_fwd", lfm_rk4_march_forward(s_, s_.vb[0], dt));
        P("midpoints", lfm_compute_midpoints(s_));
        P("viscous", lfm_compute_viscous(s_, A, mu));
        P("path_int", lfm_accumulate_path_integral(s_, u0, dt / rho));
    }

    // ── Steps 6-10: second midpoint u_{3/2} (= B) ──
    if (n_steps >= 2) {
        if (mu > 0) {
            P("viscous", lfm_compute_viscous(s_, A, mu));
            P("accum", lfm_accumulate_to_u0(s_, A, s_.visc_x, s_.visc_y, s_.visc_z, dt / rho));
        }
        P("copy", lfm_copy_vel(s_, B, A));
        P("advect", lfm_rk2_advect(s_, B, A, s_.vb[0], dt));
        P("bc", apply_bc(B));
        P("project", lfm_project(s_, B, dt, iters, tol));
        P("bc", apply_bc(B));
        P("copy", lfm_copy_vel(s_, s_.vb[1], B));
        if (mu > 0) {
            P("save_fm", lfm_save_flow_map_state(s_));
            P("march_fwd", lfm_rk4_march_forward(s_, s_.vb[1], dt));
            P("midpoints", lfm_compute_midpoints(s_));
            P("viscous", lfm_compute_viscous(s_, B, mu));
            P("path_int", lfm_accumulate_path_integral(s_, u0, dt / rho));
        }
    }

    // ── Steps 11-17: main leapfrog loop (sliding window im32/im12/nxt) ──
    CudaVel3D im32 = A, im12 = B, nxt = C;
    P("copy", lfm_copy_vel(s_, im32, s_.vb[0]));
    if (n_steps >= 2)
        P("copy", lfm_copy_vel(s_, im12, s_.vb[1]));
    for (int i_step = 2; i_step < n_steps; i_step++) {
        if (mu > 0) {
            P("viscous", lfm_compute_viscous(s_, im32, mu));
            P("accum", lfm_accumulate_to_u0(s_, im32, s_.visc_x, s_.visc_y, s_.visc_z, 2.0 * dt / rho));
        }
        P("copy", lfm_copy_vel(s_, nxt, im32));
        P("advect", lfm_rk2_advect(s_, nxt, im32, s_.vb[i_step - 1], 2.0 * dt));
        P("bc", apply_bc(nxt));
        P("project", lfm_project(s_, nxt, dt, iters, tol));
        P("bc", apply_bc(nxt));
        P("copy", lfm_copy_vel(s_, s_.vb[i_step], nxt));
        if (mu > 0) {
            P("save_fm", lfm_save_flow_map_state(s_));
            P("march_fwd", lfm_rk4_march_forward(s_, s_.vb[i_step], dt));
            P("midpoints", lfm_compute_midpoints(s_));
            P("viscous", lfm_compute_viscous(s_, nxt, mu));
            P("path_int", lfm_accumulate_path_integral(s_, u0, dt / rho));
        }
        CudaVel3D t = im32;
        im32        = im12;
        im12        = nxt;
        nxt         = t; // slide window
    }

    // ── Steps 18-26 (FIX①): per-axis (staggered-face) flow maps ──
    // Forward face map integrated forward over the buffer, backward face map
    // integrated backward over the reversed buffer (mirrors the CPU + author).
    P("face_identity", lfm_face_set_forward_identity(s_));
    for (int i_step = 1; i_step <= n_steps; i_step++)
        P("face_march", lfm_face_march_forward(s_, s_.vb[i_step - 1], dt));
    P("face_identity", lfm_face_set_backward_identity(s_));
    for (int i_step = n_steps; i_step >= 1; i_step--)
        P("face_march", lfm_face_march_backward(s_, s_.vb[i_step - 1], -dt));

    // ── Step 22: per-face pullback m_a = T_a · u0(ψ_a) → mface ──
    P("face_pullback", lfm_face_pullback(s_, u0, s_.mface, /*fwd=*/false));

    // ── Steps 23-27: face BFECC + write mface → cur directly (no gauge avg) ──
    P("face_bfecc", lfm_face_error_correction(s_, u0, cur, cfg_.lfm_bfecc_clamp));

    double cycle_dt = n_steps * dt;
    P("project_end", lfm_project(s_, cur, cycle_dt, iters * 2, tol));
    P("bc", apply_bc(cur));

    t_ += n_steps * dt;
    step_++;
    sync_to_host();
}

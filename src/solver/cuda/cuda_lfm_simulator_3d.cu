// ════════════════════════════════════════════════════════════════════
// Host orchestration of the fully GPU-resident 3D LFM cycle. Mirrors
// LFMSimulator3D::run_cycle (src/simulator/lfm_simulator_3d.cpp) line-for-line,
// but every sub-step is a kernel launch / device Poisson solve — the field
// never leaves the GPU until sync_to_host().
// ════════════════════════════════════════════════════════════════════
#include "simulator/cuda_lfm_simulator_3d.h"
#include <algorithm>
#include <vector>

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
    lfm_copy_vel(s_, u0, cur);
    if (mu > 0) {
        lfm_compute_viscous(s_, u0, mu);
        lfm_accumulate_to_u0(s_, u0, s_.visc_x, s_.visc_y, s_.visc_z, dt / (2.0 * rho));
    }

    lfm_set_identity(s_);

    // ── Steps 2-5: first midpoint u_{1/2} (= A) ──
    lfm_copy_vel(s_, A, cur);
    lfm_rk2_advect(s_, A, cur, cur, dt / 2.0);
    apply_bc(A);
    lfm_project(s_, A, dt, iters, tol);
    apply_bc(A);
    lfm_copy_vel(s_, s_.vb[0], A);
    lfm_save_flow_map_state(s_);
    lfm_rk4_march_forward(s_, s_.vb[0], dt);
    lfm_compute_midpoints(s_);
    if (mu > 0) {
        lfm_compute_viscous(s_, A, mu);
        lfm_accumulate_path_integral(s_, u0, dt / rho);
    }

    // ── Steps 6-10: second midpoint u_{3/2} (= B) ──
    if (n_steps >= 2) {
        if (mu > 0) {
            lfm_compute_viscous(s_, A, mu);
            lfm_accumulate_to_u0(s_, A, s_.visc_x, s_.visc_y, s_.visc_z, dt / rho); // u_{1/2}^†
        }
        lfm_copy_vel(s_, B, A);
        lfm_rk2_advect(s_, B, A, s_.vb[0], dt);
        apply_bc(B);
        lfm_project(s_, B, dt, iters, tol);
        apply_bc(B);
        lfm_copy_vel(s_, s_.vb[1], B);
        lfm_save_flow_map_state(s_);
        lfm_rk4_march_forward(s_, s_.vb[1], dt);
        lfm_compute_midpoints(s_);
        if (mu > 0) {
            lfm_compute_viscous(s_, B, mu);
            lfm_accumulate_path_integral(s_, u0, dt / rho);
        }
    }

    // ── Steps 11-17: main leapfrog loop (sliding window im32/im12/nxt) ──
    CudaVel3D im32 = A, im12 = B, nxt = C;
    lfm_copy_vel(s_, im32, s_.vb[0]);
    if (n_steps >= 2)
        lfm_copy_vel(s_, im12, s_.vb[1]);
    for (int i_step = 2; i_step < n_steps; i_step++) {
        if (mu > 0) {
            lfm_compute_viscous(s_, im32, mu);
            lfm_accumulate_to_u0(s_, im32, s_.visc_x, s_.visc_y, s_.visc_z, 2.0 * dt / rho);
        }
        lfm_copy_vel(s_, nxt, im32);
        lfm_rk2_advect(s_, nxt, im32, s_.vb[i_step - 1], 2.0 * dt);
        apply_bc(nxt);
        lfm_project(s_, nxt, dt, iters, tol);
        apply_bc(nxt);
        lfm_copy_vel(s_, s_.vb[i_step], nxt);
        lfm_save_flow_map_state(s_);
        lfm_rk4_march_forward(s_, s_.vb[i_step], dt);
        lfm_compute_midpoints(s_);
        if (mu > 0) {
            lfm_compute_viscous(s_, nxt, mu);
            lfm_accumulate_path_integral(s_, u0, dt / rho);
        }
        CudaVel3D t = im32;
        im32        = im12;
        im12        = nxt;
        nxt         = t; // slide window
    }

    // ── Steps 18-21: backward march ──
    lfm_set_backward_identity(s_);
    for (int i_step = n_steps; i_step >= 1; i_step--)
        lfm_rk4_march_backward(s_, s_.vb[i_step - 1], -dt);

    // ── Step 22 + 23-26: pullback + error correction ──
    lfm_pullback_impulse(s_, u0);
    lfm_error_correction(s_, u0, cfg_.lfm_bfecc_clamp);

    // ── Step 27: gauge projection u_n ← Project(m_n) ──
    lfm_gauge_writeback(s_, cur);
    double cycle_dt = n_steps * dt;
    lfm_project(s_, cur, cycle_dt, iters * 2, tol);
    apply_bc(cur);

    t_ += n_steps * dt;
    step_++;
    sync_to_host();
}

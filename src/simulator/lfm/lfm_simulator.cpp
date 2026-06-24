#include "simulator/lfm/lfm_simulator.h"
#include "numerics/pressure/pressure.h"
#include "simulator/scenarios/scenario_registry.h"
#include "io/vtk_writer.h"
#include <cmath>
#include <iostream>
#include <sys/stat.h>

LFMSimulator::LFMSimulator(const Config& cfg, std::unique_ptr<Solver> solver)
    : cfg_(cfg), grid_(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly), solver_(std::move(solver)),
      flow_map_(cfg.NX, cfg.NY, cfg.Lx / cfg.NX, cfg.Ly / cfg.NY) {
    int N = cfg_.NX * cfg_.NY;
    // FIX①: impulse + per-face flow maps live on MAC faces (u/v sized).
    size_t us = (size_t)grid_.u_size(), vs = (size_t)grid_.v_size();
    m_x_.assign(us, 0.0);
    m_y_.assign(vs, 0.0);
    auto alloc_face = [](FaceFlowMap2D& f, size_t n) {
        for (auto* p : {&f.fx, &f.fy, &f.f0, &f.f1, &f.bx, &f.by, &f.t0, &f.t1})
            p->assign(n, 0.0);
    };
    alloc_face(fmu_, us);
    alloc_face(fmv_, vs);
    phi_mid_x_.resize(N, 0.0);
    phi_mid_y_.resize(N, 0.0);
    F_mid_00_.resize(N, 0.0);
    F_mid_10_.resize(N, 0.0);
    F_mid_01_.resize(N, 0.0);
    F_mid_11_.resize(N, 0.0);

    scenario_ = scenarios::ScenarioRegistry::instance().create(cfg_.scenario);
    scenario_->init_grid(grid_, cfg_);
    bcs_ = scenario_->boundary_manager(cfg_);
    bcs_.apply(grid_);
}

void LFMSimulator::step() {
    run_cycle(cfg_.lfm_cycle_steps);
}

// ═══════════════════════════════════════════════════════════════
// Algorithm 1: LFM Reinitialization Cycle (Sun et al. SIGGRAPH 2025)
//
// Pure implementation — no artificial constraints beyond what the
// paper describes. velocity_gradient returns 0 near solid boundaries
// (central differences require two fluid neighbors), which naturally
// gives dF/dt ≈ 0 and F ≈ I near walls without any explicit override.
// ═══════════════════════════════════════════════════════════════
void LFMSimulator::run_cycle(int n_steps) {
    double dt  = cfg_.dt;
    double rho = 1.0;
    double mu  = (cfg_.Re > 0) ? cfg_.U_inf * 2.0 * cfg_.cyl_R / cfg_.Re : 0.0;
    int nx = grid_.nx, ny = grid_.ny;

    Grid u0_grid = grid_;
    if (mu > 0) {
        std::vector<double> vu, vv;
        compute_viscous(u0_grid, vu, vv);
        accumulate_to_u0(u0_grid, vu, vv, dt / (2.0 * rho)); // Step 1: half-step viscous
    }

    vel_buffer_.clear();
    vel_buffer_.resize(n_steps);
    flow_map_.set_identity();

    // ── Steps 2-5: First midpoint u_{1/2} ──
    Grid u_half(nx, ny, grid_.Lx(), grid_.Ly());
    u_half = grid_;
    rk2_advect(u_half, grid_, grid_.u, grid_.v, dt / 2.0);
    bcs_.apply(u_half);
    project(u_half);
    bcs_.apply(u_half);
    vel_buffer_[0] = {u_half.u, u_half.v};

    // Step 4: Forward march, save midpoint for Step 5 path integral
    save_flow_map_state();
    rk4_march_forward(vel_buffer_[0].u, vel_buffer_[0].v, dt);
    compute_midpoints();
    if (mu > 0) {
        std::vector<double> vu, vv;
        compute_viscous(u_half, vu, vv);
        accumulate_path_integral(u0_grid, vu, vv, dt / rho);
    }

    // ── Steps 6-10: Second midpoint u_{3/2} ──
    if (n_steps >= 2) {
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_half, vu, vv);
            accumulate_to_u0(u_half, vu, vv, dt / rho); // u_{1/2}^† (Step 6)
        }
        Grid u_3half(nx, ny, grid_.Lx(), grid_.Ly());
        u_3half = u_half;
        rk2_advect(u_3half, u_half, vel_buffer_[0].u, vel_buffer_[0].v, dt);
        bcs_.apply(u_3half);
        project(u_3half);
        bcs_.apply(u_3half);
        vel_buffer_[1] = {u_3half.u, u_3half.v};

        save_flow_map_state();
        rk4_march_forward(vel_buffer_[1].u, vel_buffer_[1].v, dt);
        compute_midpoints();
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_3half, vu, vv);
            accumulate_path_integral(u0_grid, vu, vv, dt / rho);
        }
    }

    // ── Steps 11-17: Main loop i=2..n-1 (leapfrog) ──
    // Source for advection AND viscosity is u_{i-3/2}; advection velocity is u_{i-1/2}.
    // Use vel_buffer_ entries (original, unmodified midpoint velocities).
    Grid u_im32 = grid_; // template (keeps solid mask)
    Grid u_im12 = grid_;
    u_im32.u    = vel_buffer_[0].u;
    u_im32.v    = vel_buffer_[0].v; // u_{1/2}
    if (n_steps >= 2) {
        u_im12.u = vel_buffer_[1].u;
        u_im12.v = vel_buffer_[1].v; // u_{3/2}
    }
    for (int i_step = 2; i_step < n_steps; i_step++) {
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_im32, vu, vv);
            accumulate_to_u0(u_im32, vu, vv, 2.0 * dt / rho); // u_{i-3/2}^† (Step 12)
        }
        Grid u_next(nx, ny, grid_.Lx(), grid_.Ly());
        u_next = u_im32;
        // Leapfrog: advect u_{i-3/2}^† with velocity field u_{i-1/2} for 2Δt
        rk2_advect(u_next, u_im32, vel_buffer_[i_step - 1].u, vel_buffer_[i_step - 1].v, 2.0 * dt);
        bcs_.apply(u_next);
        project(u_next);
        bcs_.apply(u_next);
        vel_buffer_[i_step] = {u_next.u, u_next.v};

        save_flow_map_state();
        rk4_march_forward(vel_buffer_[i_step].u, vel_buffer_[i_step].v, dt);
        compute_midpoints();
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_next, vu, vv);
            accumulate_path_integral(u0_grid, vu, vv, dt / rho);
        }
        // Slide window: u_{i-3/2} ← u_{i-1/2}, u_{i-1/2} ← u_{i+1/2}
        u_im32 = u_im12;
        u_im12 = u_next;
    }

    // ── Steps 18-27 (FIX①): per-axis (staggered-face) flow maps ──
    // Build the forward face map (φ_a, F_a) by integrating forward over the
    // velocity buffer, and the backward face map (ψ_a, T_a) by integrating
    // backward over the reversed buffer — mirroring the author's per-axis
    // ReinitAsync. The impulse then lands directly on the MAC faces (no avg).
    (void)nx;
    (void)ny;
    face_set_forward_identity();
    for (int i_step = 1; i_step <= n_steps; i_step++)
        face_march_forward(vel_buffer_[i_step - 1].u, vel_buffer_[i_step - 1].v, dt);
    face_set_backward_identity();
    for (int i_step = n_steps; i_step >= 1; i_step--)
        face_march_backward(vel_buffer_[i_step - 1].u, vel_buffer_[i_step - 1].v, -dt);

    // ── Step 22: per-face pullback m_a = T_a · u0(ψ_a) ──
    face_pullback(u0_grid.u, u0_grid.v, m_x_, m_y_, /*fwd=*/false);

    // ── Steps 23-27: face BFECC error correction + write m → grid (no gauge avg) ──
    face_error_correction(u0_grid);

    double cycle_dt = n_steps * dt;
    PressureProjection::project(grid_, cycle_dt, *solver_, cfg_.solve_iters * 2, cfg_.solve_tol);
    // The projection's scalar field is the gauge potential, NOT the static
    // pressure. Recover the true static pressure from the divergence-free
    // velocity via the pressure Poisson equation, so force diagnostics (Cd/Cl)
    // integrate the correct surface pressure.
    PressureProjection::recoverStaticPressure(grid_, *solver_, cfg_.solve_iters * 2,
                                              cfg_.solve_tol);
    bcs_.apply(grid_);

    t_ += n_steps * dt;
    step_++;
}

// ═══════════════════════════════════════════════════════════
// Midpoint state for path integral: saved before each RK4 step
// ═══════════════════════════════════════════════════════════
void LFMSimulator::save_flow_map_state() {
    std::copy(flow_map_.phi_x.begin(), flow_map_.phi_x.end(), phi_mid_x_.begin());
    std::copy(flow_map_.phi_y.begin(), flow_map_.phi_y.end(), phi_mid_y_.begin());
    std::copy(flow_map_.F00.begin(), flow_map_.F00.end(), F_mid_00_.begin());
    std::copy(flow_map_.F10.begin(), flow_map_.F10.end(), F_mid_10_.begin());
    std::copy(flow_map_.F01.begin(), flow_map_.F01.end(), F_mid_01_.begin());
    std::copy(flow_map_.F11.begin(), flow_map_.F11.end(), F_mid_11_.begin());
}

void LFMSimulator::compute_midpoints() {
    int N = (int)flow_map_.phi_x.size();
    for (int k = 0; k < N; k++) {
        phi_mid_x_[k] = 0.5 * (phi_mid_x_[k] + flow_map_.phi_x[k]);
        phi_mid_y_[k] = 0.5 * (phi_mid_y_[k] + flow_map_.phi_y[k]);
        F_mid_00_[k]  = 0.5 * (F_mid_00_[k] + flow_map_.F00[k]);
        F_mid_10_[k]  = 0.5 * (F_mid_10_[k] + flow_map_.F10[k]);
        F_mid_01_[k]  = 0.5 * (F_mid_01_[k] + flow_map_.F01[k]);
        F_mid_11_[k]  = 0.5 * (F_mid_11_[k] + flow_map_.F11[k]);
    }
}

// ═══════════════════════════════════════════════════════════
// u_0 += coeff * (cell-centered viscous force averaged to faces).
// Adds to face values directly so the MAC layout is preserved.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::accumulate_to_u0(Grid& u0, const std::vector<double>& vu,
                                    const std::vector<double>& vv, double coeff) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i < nx; i++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i + 1, j))
                continue;
            u0.u_at(i, j) += coeff * 0.5 * (vu[flow_map_.idx(i, j)] + vu[flow_map_.idx(i + 1, j)]);
        }
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j < ny; j++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i, j + 1))
                continue;
            u0.v_at(i, j) += coeff * 0.5 * (vv[flow_map_.idx(i, j)] + vv[flow_map_.idx(i, j + 1)]);
        }
}

// ═══════════════════════════════════════════════════════════
// Path integral: u_0 += coeff * F_mid^T · visc(Φ_mid).
// Computes cell-centered contribution first, then averages to faces.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::accumulate_path_integral(Grid& u0, const std::vector<double>& visc_u,
                                            const std::vector<double>& visc_v, double coeff) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
    std::vector<double> cx(nx * ny, 0.0), cy(nx * ny, 0.0);
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx; i++) {
            if (grid_.is_solid(i, j))
                continue;
            size_t k = flow_map_.idx(i, j);
            double vx, vy;
            sample_cell_centered(visc_u, visc_v, phi_mid_x_[k], phi_mid_y_[k], vx, vy);
            cx[k] = F_mid_00_[k] * vx + F_mid_10_[k] * vy;
            cy[k] = F_mid_01_[k] * vx + F_mid_11_[k] * vy;
        }
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i < nx; i++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i + 1, j))
                continue;
            u0.u_at(i, j) += coeff * 0.5 * (cx[flow_map_.idx(i, j)] + cx[flow_map_.idx(i + 1, j)]);
        }
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j < ny; j++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i, j + 1))
                continue;
            u0.v_at(i, j) += coeff * 0.5 * (cy[flow_map_.idx(i, j)] + cy[flow_map_.idx(i, j + 1)]);
        }
}

// ═══════════════════════════════════════════════════════════
// RK2 semi-Lagrangian advection
// ═══════════════════════════════════════════════════════════
void LFMSimulator::rk2_advect(Grid& dst, const Grid& src, const std::vector<double>& vel_u,
                              const std::vector<double>& vel_v, double dt_step) {
    int nx = grid_.nx, ny = grid_.ny;
    double dx = grid_.dx, dy = grid_.dy;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < nx; i++)
        for (int j = 1; j <= ny; j++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i + 1, j)) {
                dst.u_at(i, j) = 0;
                continue;
            }
            double xu = i * dx, yu = (j - 0.5) * dy;
            double u1 = sample_u(xu, yu, vel_u, vel_v), v1 = sample_v(xu, yu, vel_u, vel_v);
            double xm = xu - 0.5 * dt_step * u1, ym = yu - 0.5 * dt_step * v1;
            double um = sample_u(xm, ym, vel_u, vel_v), vm = sample_v(xm, ym, vel_u, vel_v);
            dst.u_at(i, j) = sample_u(xu - dt_step * um, yu - dt_step * vm, src.u, src.v);
        }
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j < ny; j++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i, j + 1)) {
                dst.v_at(i, j) = 0;
                continue;
            }
            double xv = (i - 0.5) * dx, yv = j * dy;
            double u1 = sample_u(xv, yv, vel_u, vel_v), v1 = sample_v(xv, yv, vel_u, vel_v);
            double xm = xv - 0.5 * dt_step * u1, ym = yv - 0.5 * dt_step * v1;
            double um = sample_u(xm, ym, vel_u, vel_v), vm = sample_v(xm, ym, vel_u, vel_v);
            dst.v_at(i, j) = sample_v(xv - dt_step * um, yv - dt_step * vm, src.u, src.v);
        }
}

void LFMSimulator::project(Grid& g) {
    PressureProjection::project(g, cfg_.dt, *solver_, cfg_.solve_iters, cfg_.solve_tol);
}

// ═══════════════════════════════════════════════════════════
// RK4-March forward: dΦ/dt = u(Φ), dF/dt = ∇u(Φ)·F
// No clamps, no near-solid override — velocity_gradient returns 0
// near solids, which naturally gives dF/dt=0 and F≈I at boundaries.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::rk4_march_forward(const std::vector<double>& u, const std::vector<double>& v,
                                     double dt_march) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx; i++) {
            size_t k  = flow_map_.idx(i, j);
            double x0 = flow_map_.phi_x[k], y0 = flow_map_.phi_y[k];
            double f00 = flow_map_.F00[k], f10 = flow_map_.F10[k];
            double f01 = flow_map_.F01[k], f11 = flow_map_.F11[k];

            auto rhs = [&](double px, double py, double c00, double c10, double c01, double c11,
                           double& dx, double& dy, double& d00, double& d10, double& d01,
                           double& d11) {
                double vu, vv;
                sample_velocity(px, py, u, v, vu, vv);
                dx = vu;
                dy = vv;
                double dudx, dudy, dvdx, dvdy;
                velocity_gradient_at(px, py, u, v, dudx, dudy, dvdx, dvdy);
                d00 = dudx * c00 + dudy * c10;
                d10 = dvdx * c00 + dvdy * c10;
                d01 = dudx * c01 + dudy * c11;
                d11 = dvdx * c01 + dvdy * c11;
            };

            double k1[6], k2[6], k3[6], k4[6];
            rhs(x0, y0, f00, f10, f01, f11, k1[0], k1[1], k1[2], k1[3], k1[4], k1[5]);
            for (int m = 0; m < 6; m++)
                k1[m] *= dt_march;
            rhs(x0 + 0.5 * k1[0], y0 + 0.5 * k1[1], f00 + 0.5 * k1[2], f10 + 0.5 * k1[3],
                f01 + 0.5 * k1[4], f11 + 0.5 * k1[5], k2[0], k2[1], k2[2], k2[3], k2[4], k2[5]);
            for (int m = 0; m < 6; m++)
                k2[m] *= dt_march;
            rhs(x0 + 0.5 * k2[0], y0 + 0.5 * k2[1], f00 + 0.5 * k2[2], f10 + 0.5 * k2[3],
                f01 + 0.5 * k2[4], f11 + 0.5 * k2[5], k3[0], k3[1], k3[2], k3[3], k3[4], k3[5]);
            for (int m = 0; m < 6; m++)
                k3[m] *= dt_march;
            rhs(x0 + k3[0], y0 + k3[1], f00 + k3[2], f10 + k3[3], f01 + k3[4], f11 + k3[5], k4[0],
                k4[1], k4[2], k4[3], k4[4], k4[5]);
            for (int m = 0; m < 6; m++)
                k4[m] *= dt_march;

            flow_map_.phi_x[k] = std::max(
                0.0, std::min(grid_.Lx(), x0 + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0));
            flow_map_.phi_y[k] = std::max(
                0.0, std::min(grid_.Ly(), y0 + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0));
            double nF00 = f00 + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0;
            double nF10 = f10 + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0;
            double nF01 = f01 + (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6.0;
            double nF11 = f11 + (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) / 6.0;
            if (!std::isfinite(nF00))
                nF00 = 1.0;
            if (!std::isfinite(nF10))
                nF10 = 0.0;
            if (!std::isfinite(nF01))
                nF01 = 0.0;
            if (!std::isfinite(nF11))
                nF11 = 1.0;
            flow_map_.F00[k] = nF00;
            flow_map_.F10[k] = nF10;
            flow_map_.F01[k] = nF01;
            flow_map_.F11[k] = nF11;
        }
}

// ═══════════════════════════════════════════════════════════
// RK4-March backward: dΨ/dt = u(Ψ), dT/dt = +∇u(Ψ)·T
// Note: with dt_march = -dt, this gives T ≈ F^{-1} (correct inverse).
// The paper has dT/dt = -∇u·T, but that combined with -dt step
// gives T ≈ F instead of T ≈ F^{-1}. We fix the sign here.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::rk4_march_backward(const std::vector<double>& u, const std::vector<double>& v,
                                      double dt_march) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx; i++) {
            size_t k  = flow_map_.idx(i, j);
            double x0 = flow_map_.psi_x[k], y0 = flow_map_.psi_y[k];
            double t00 = flow_map_.T00[k], t10 = flow_map_.T10[k];
            double t01 = flow_map_.T01[k], t11 = flow_map_.T11[k];

            auto rhs = [&](double px, double py, double c00, double c10, double c01, double c11,
                           double& dx, double& dy, double& d00, double& d10, double& d01,
                           double& d11) {
                double vu, vv;
                sample_velocity(px, py, u, v, vu, vv);
                dx = vu;
                dy = vv;
                double dudx, dudy, dvdx, dvdy;
                velocity_gradient_at(px, py, u, v, dudx, dudy, dvdx, dvdy);
                d00 = dudx * c00 + dudy * c10;
                d10 = dvdx * c00 + dvdy * c10;
                d01 = dudx * c01 + dudy * c11;
                d11 = dvdx * c01 + dvdy * c11;
            };

            double k1[6], k2[6], k3[6], k4[6];
            rhs(x0, y0, t00, t10, t01, t11, k1[0], k1[1], k1[2], k1[3], k1[4], k1[5]);
            for (int m = 0; m < 6; m++)
                k1[m] *= dt_march;
            rhs(x0 + 0.5 * k1[0], y0 + 0.5 * k1[1], t00 + 0.5 * k1[2], t10 + 0.5 * k1[3],
                t01 + 0.5 * k1[4], t11 + 0.5 * k1[5], k2[0], k2[1], k2[2], k2[3], k2[4], k2[5]);
            for (int m = 0; m < 6; m++)
                k2[m] *= dt_march;
            rhs(x0 + 0.5 * k2[0], y0 + 0.5 * k2[1], t00 + 0.5 * k2[2], t10 + 0.5 * k2[3],
                t01 + 0.5 * k2[4], t11 + 0.5 * k2[5], k3[0], k3[1], k3[2], k3[3], k3[4], k3[5]);
            for (int m = 0; m < 6; m++)
                k3[m] *= dt_march;
            rhs(x0 + k3[0], y0 + k3[1], t00 + k3[2], t10 + k3[3], t01 + k3[4], t11 + k3[5], k4[0],
                k4[1], k4[2], k4[3], k4[4], k4[5]);
            for (int m = 0; m < 6; m++)
                k4[m] *= dt_march;

            flow_map_.psi_x[k] = std::max(
                0.0, std::min(grid_.Lx(), x0 + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0));
            flow_map_.psi_y[k] = std::max(
                0.0, std::min(grid_.Ly(), y0 + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0));
            double nT00 = t00 + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0;
            double nT10 = t10 + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0;
            double nT01 = t01 + (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6.0;
            double nT11 = t11 + (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) / 6.0;
            if (!std::isfinite(nT00))
                nT00 = 1.0;
            if (!std::isfinite(nT10))
                nT10 = 0.0;
            if (!std::isfinite(nT01))
                nT01 = 0.0;
            if (!std::isfinite(nT11))
                nT11 = 1.0;
            flow_map_.T00[k] = nT00;
            flow_map_.T10[k] = nT10;
            flow_map_.T01[k] = nT01;
            flow_map_.T11[k] = nT11;
        }
}

// ═══════════════════════════════════════════════════════════
// Quadratic B-spline weights (paper §4.1): three-point kernel.
// r ∈ [-0.5, 0.5] is the offset from the nearest grid point.
// w[0] for (nearest-1), w[1] for nearest, w[2] for (nearest+1).
// Sum is 1 by construction. C¹ smooth → much less spurious dissipation
// than bilinear for long-range flow-map sampling.
// ═══════════════════════════════════════════════════════════
static inline void bspline_weights(double r, double w[3]) {
    double a = 0.5 - r;
    double b = 0.5 + r;
    w[0]     = 0.5 * a * a;
    w[1]     = 0.75 - r * r;
    w[2]     = 0.5 * b * b;
}

// ═══════════════════════════════════════════════════════════
// Velocity interpolation (quadratic B-spline, MAC-grid aware)
// ═══════════════════════════════════════════════════════════
void LFMSimulator::sample_velocity(double x, double y, const std::vector<double>& u_vec,
                                   const std::vector<double>& v_vec, double& vu, double& vv) const {
    x         = std::max(0.0, std::min(grid_.Lx(), x));
    y         = std::max(0.0, std::min(grid_.Ly(), y));
    double dx = grid_.dx, dy = grid_.dy;
    int nx = grid_.nx, ny = grid_.ny;

    // ── u-face value: u_vec[iu(i,j)] sits at (i·dx, (j-0.5)·dy) ──
    {
        double ux = x / dx;       // ∈ [0, nx]
        double uy = y / dy + 0.5; // ∈ [0.5, ny+0.5]
        int ic    = (int)std::floor(ux + 0.5);
        int jc    = (int)std::floor(uy + 0.5);
        double wx[3], wy[3];
        bspline_weights(ux - ic, wx);
        bspline_weights(uy - jc, wy);
        auto u_at = [&](int ii, int jj) -> double {
            ii = std::max(0, std::min(nx, ii));
            jj = std::max(1, std::min(ny, jj));
            return u_vec[ii + jj * (nx + 1)];
        };
        vu = 0.0;
        for (int dj = -1; dj <= 1; dj++)
            for (int di = -1; di <= 1; di++)
                vu += wx[di + 1] * wy[dj + 1] * u_at(ic + di, jc + dj);
    }

    // ── v-face value: v_vec[iv(i,j)] sits at ((i-0.5)·dx, j·dy) ──
    {
        double vx = x / dx + 0.5; // ∈ [0.5, nx+0.5]
        double vy = y / dy;       // ∈ [0, ny]
        int ic    = (int)std::floor(vx + 0.5);
        int jc    = (int)std::floor(vy + 0.5);
        double wx[3], wy[3];
        bspline_weights(vx - ic, wx);
        bspline_weights(vy - jc, wy);
        auto v_at = [&](int ii, int jj) -> double {
            ii = std::max(1, std::min(nx, ii));
            jj = std::max(0, std::min(ny, jj));
            return v_vec[ii + jj * (nx + 2)];
        };
        vv = 0.0;
        for (int dj = -1; dj <= 1; dj++)
            for (int di = -1; di <= 1; di++)
                vv += wx[di + 1] * wy[dj + 1] * v_at(ic + di, jc + dj);
    }
}

double LFMSimulator::sample_u(double x, double y, const std::vector<double>& u,
                              const std::vector<double>& v) const {
    double vu, vv;
    sample_velocity(x, y, u, v, vu, vv);
    return vu;
}
double LFMSimulator::sample_v(double x, double y, const std::vector<double>& u,
                              const std::vector<double>& v) const {
    double vu, vv;
    sample_velocity(x, y, u, v, vu, vv);
    return vv;
}

// Quadratic B-spline at cell centers ((i-0.5)·dx, (j-0.5)·dy), i,j ∈ [1, nx]×[1, ny].
void LFMSimulator::sample_cell_centered(const std::vector<double>& sx,
                                        const std::vector<double>& sy, double x, double y,
                                        double& vx, double& vy) const {
    x         = std::max(0.0, std::min(grid_.Lx(), x));
    y         = std::max(0.0, std::min(grid_.Ly(), y));
    double dx = grid_.dx, dy = grid_.dy;
    int nx = grid_.nx, ny = grid_.ny;
    double cix = x / dx + 0.5; // ∈ [0.5, nx+0.5]
    double ciy = y / dy + 0.5;
    int ic     = (int)std::floor(cix + 0.5);
    int jc     = (int)std::floor(ciy + 0.5);
    double wx[3], wy[3];
    bspline_weights(cix - ic, wx);
    bspline_weights(ciy - jc, wy);
    vx = 0.0;
    vy = 0.0;
    for (int dj = -1; dj <= 1; dj++) {
        int jj = std::max(1, std::min(ny, jc + dj));
        for (int di = -1; di <= 1; di++) {
            int ii   = std::max(1, std::min(nx, ic + di));
            double w = wx[di + 1] * wy[dj + 1];
            size_t k = flow_map_.idx(ii, jj);
            vx += w * sx[k];
            vy += w * sy[k];
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Velocity gradient at arbitrary position → nearest cell center
// ═══════════════════════════════════════════════════════════
void LFMSimulator::velocity_gradient_at(double x, double y, const std::vector<double>& ug,
                                        const std::vector<double>& vg, double& du_dx, double& du_dy,
                                        double& dv_dx, double& dv_dy) const {
    int ci = std::max(1, std::min(grid_.nx, (int)(x / grid_.dx + 0.5)));
    int cj = std::max(1, std::min(grid_.ny, (int)(y / grid_.dy + 0.5)));
    velocity_gradient(ci, cj, ug, vg, du_dx, du_dy, dv_dx, dv_dy);
}

// ═══════════════════════════════════════════════════════════
// Velocity gradient at cell center.
// Diagonals du_dx, dv_dy use cell's OWN two faces (no neighbor needed,
// no-slip face value is a legitimate boundary value → keep the shear).
// Off-diagonals du_dy, dv_dx need central differences across neighbors;
// zero them when either neighbor is solid (stair-step protection).
// ═══════════════════════════════════════════════════════════
void LFMSimulator::velocity_gradient(int i, int j, const std::vector<double>& ug,
                                     const std::vector<double>& vg, double& du_dx, double& du_dy,
                                     double& dv_dx, double& dv_dy) const {
    double dx = grid_.dx, dy = grid_.dy;
    int nx = grid_.nx, ny = grid_.ny;
    int ip1 = std::min(i + 1, nx), im1 = std::max(i - 1, 1);
    int jp1 = std::min(j + 1, ny), jm1 = std::max(j - 1, 1);
    auto iu = [&](int ii, int jj) { return ii + jj * (nx + 1); };
    auto iv = [&](int ii, int jj) { return ii + jj * (nx + 2); };

    bool solid_B = grid_.is_solid(i, j - 1), solid_T = grid_.is_solid(i, j + 1);
    bool solid_L = grid_.is_solid(i - 1, j), solid_R = grid_.is_solid(i + 1, j);

    du_dx = (ug[iu(i, j)] - ug[iu(i - 1, j)]) / dx;
    dv_dy = (vg[iv(i, j)] - vg[iv(i, j - 1)]) / dy;
    du_dy = (solid_B || solid_T) ? 0.0 : (ug[iu(i, jp1)] - ug[iu(i, jm1)]) / (2 * dy);
    dv_dx = (solid_L || solid_R) ? 0.0 : (vg[iv(ip1, j)] - vg[iv(im1, j)]) / (2 * dx);
}

// ═══════════════════════════════════════════════════════════
// Viscous force μ∇²u at cell centers
// ═══════════════════════════════════════════════════════════
void LFMSimulator::compute_viscous(const Grid& g, std::vector<double>& vu,
                                   std::vector<double>& vv) {
    int nx = g.nx, ny = g.ny;
    double mu   = (cfg_.Re > 0) ? cfg_.U_inf * 2 * cfg_.cyl_R / cfg_.Re : 0;
    double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy);
    size_t N = flow_map_.nx * flow_map_.ny;
    vu.assign(N, 0);
    vv.assign(N, 0);
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx; i++) {
            size_t k = flow_map_.idx(i, j);
            if (g.is_solid(i, j))
                continue;
            double uc = 0.5 * (g.u_at(i, j) + g.u_at(i - 1, j));
            double uL = (i > 1) ? 0.5 * (g.u_at(i - 1, j) + g.u_at(i - 2, j)) : uc;
            double uR = (i < nx) ? 0.5 * (g.u_at(i + 1, j) + g.u_at(i, j)) : uc;
            double uB = (j > 1) ? 0.5 * (g.u_at(i, j - 1) + g.u_at(i - 1, j - 1)) : uc;
            double uT = (j < ny) ? 0.5 * (g.u_at(i, j + 1) + g.u_at(i - 1, j + 1)) : uc;
            vu[k]     = mu * ((uL + uR - 2 * uc) * idx2 + (uB + uT - 2 * uc) * idy2);
            double vc = 0.5 * (g.v_at(i, j) + g.v_at(i, j - 1));
            double vL = (i > 1) ? 0.5 * (g.v_at(i - 1, j) + g.v_at(i - 1, j - 1)) : vc;
            double vR = (i < nx) ? 0.5 * (g.v_at(i + 1, j) + g.v_at(i + 1, j - 1)) : vc;
            double vB = (j > 1) ? 0.5 * (g.v_at(i, j - 1) + g.v_at(i, j - 2)) : vc;
            double vT = (j < ny) ? 0.5 * (g.v_at(i, j + 1) + g.v_at(i, j)) : vc;
            vv[k]     = mu * ((vL + vR - 2 * vc) * idx2 + (vB + vT - 2 * vc) * idy2);
        }
}

// ═══════════════════════════════════════════════════════════
// Pullback: m = T^T · u_0(Ψ)
// ═══════════════════════════════════════════════════════════
void LFMSimulator::pullback_impulse(const Grid& u0_grid) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx; i++) {
            size_t k = flow_map_.idx(i, j);
            double X = flow_map_.psi_x[k], Y = flow_map_.psi_y[k], uX, uY;
            sample_velocity(X, Y, u0_grid.u, u0_grid.v, uX, uY);
            m_x_[k] = flow_map_.T00[k] * uX + flow_map_.T10[k] * uY;
            m_y_[k] = flow_map_.T01[k] * uX + flow_map_.T11[k] * uY;
        }
}

// ═══════════════════════════════════════════════════════════
// Forward pullback: û_0 = F^T · m(Φ)   — uses B-spline via sample_cell_centered.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::forward_pullback(const std::vector<double>& mx, const std::vector<double>& my,
                                    std::vector<double>& ux, std::vector<double>& uy) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx; i++) {
            size_t k = flow_map_.idx(i, j);
            double msx, msy;
            sample_cell_centered(mx, my, flow_map_.phi_x[k], flow_map_.phi_y[k], msx, msy);
            ux[k] = flow_map_.F00[k] * msx + flow_map_.F10[k] * msy;
            uy[k] = flow_map_.F01[k] * msx + flow_map_.F11[k] * msy;
        }
}

// ═══════════════════════════════════════════════════════════════════════
// d/dr of the quadratic B-spline weights (analytic spline derivative ≡ 3D dN2).
// ═══════════════════════════════════════════════════════════════════════
static inline void dbspline_weights(double r, double dw[3]) {
    dw[0] = r - 0.5;  // d/dr[0.5(0.5-r)^2]
    dw[1] = -2.0 * r; // d/dr[0.75-r^2]
    dw[2] = r + 0.5;  // d/dr[0.5(0.5+r)^2]
}

// Velocity + analytic 2x2 B-spline gradient (2D reduction of the verified 3D
// sample_velocity_gradient): same B-spline as sample_velocity, so the marched
// ∇u is consistent with the sampled u. g[2a+b] = ∂u_a/∂x_b.
void LFMSimulator::sample_velocity_gradient(double x, double y, const std::vector<double>& u_vec,
                                            const std::vector<double>& v_vec, double& vu, double& vv,
                                            double g[4]) const {
    x         = std::max(0.0, std::min(grid_.Lx(), x));
    y         = std::max(0.0, std::min(grid_.Ly(), y));
    double dx = grid_.dx, dy = grid_.dy;
    int nx = grid_.nx, ny = grid_.ny;

    auto gather = [&](double cx, double cy, auto at, double& val, double& gx, double& gy) {
        int ic = (int)std::floor(cx + 0.5);
        int jc = (int)std::floor(cy + 0.5);
        double wx[3], wy[3], dwx[3], dwy[3];
        bspline_weights(cx - ic, wx);
        dbspline_weights(cx - ic, dwx);
        bspline_weights(cy - jc, wy);
        dbspline_weights(cy - jc, dwy);
        val = gx = gy = 0.0;
        for (int dj = -1; dj <= 1; dj++)
            for (int di = -1; di <= 1; di++) {
                double f = at(ic + di, jc + dj);
                val += wx[di + 1] * wy[dj + 1] * f;
                gx += dwx[di + 1] * wy[dj + 1] * f;
                gy += wx[di + 1] * dwy[dj + 1] * f;
            }
    };
    double s, sx, sy;
    // u-face → row a=0; u_vec[iu(i,j)] sits at (i·dx,(j-0.5)·dy)
    gather(x / dx, y / dy + 0.5,
           [&](int ii, int jj) {
               ii = std::max(0, std::min(nx, ii));
               jj = std::max(1, std::min(ny, jj));
               return u_vec[ii + jj * (nx + 1)];
           },
           s, sx, sy);
    vu   = s;
    g[0] = sx / dx;
    g[1] = sy / dy;
    // v-face → row a=1; v_vec[iv(i,j)] sits at ((i-0.5)·dx, j·dy)
    gather(x / dx + 0.5, y / dy,
           [&](int ii, int jj) {
               ii = std::max(1, std::min(nx, ii));
               jj = std::max(0, std::min(ny, jj));
               return v_vec[ii + jj * (nx + 2)];
           },
           s, sx, sy);
    vv   = s;
    g[2] = sx / dx;
    g[3] = sy / dy;
}

// ═══════════════════════════════════════════════════════════════════════
// FIX① — per-axis (staggered-face) flow maps (2D port of the verified 3D
// FaceFlowMap code / author RKAxisKernel + PullbackAxisKernel). The impulse
// rides directly on the MAC faces: per-axis position ψ_a + Jacobian covector
// row T_a, marched with the same RK4/∇u kernel carrying only the 2-component
// covector. Pullback m_a = T_a·u0(ψ_a) lands straight on faces with ZERO
// averaging — removing the ½(a+b) gauge filters that were the dominant error.
// ═══════════════════════════════════════════════════════════════════════

// Physical coordinate of face (i,j) for the given axis (0=u, 1=v).
static inline void face_coord(int axis, int i, int j, double dx, double dy, double& x, double& y) {
    if (axis == 0) { // u-face at (i·dx,(j-0.5)·dy)
        x = i * dx;
        y = (j - 0.5) * dy;
    } else { // v-face at ((i-0.5)·dx, j·dy)
        x = (i - 0.5) * dx;
        y = j * dy;
    }
}

void LFMSimulator::face_set_forward_identity() {
    int nx = grid_.nx, ny = grid_.ny;
    double dx = grid_.dx, dy = grid_.dy;
    auto init = [&](FaceFlowMap2D& f, int axis, int imax, int jmax, auto idx) {
        for (int j = 1; j <= jmax; j++)
            for (int i = 1; i <= imax; i++) {
                int m = idx(i, j);
                double x, y;
                face_coord(axis, i, j, dx, dy, x, y);
                f.fx[m] = x;
                f.fy[m] = y;
                f.f0[m] = (axis == 0) ? 1.0 : 0.0;
                f.f1[m] = (axis == 1) ? 1.0 : 0.0;
            }
    };
    init(fmu_, 0, nx - 1, ny, [&](int i, int j) { return grid_.iu(i, j); });
    init(fmv_, 1, nx, ny - 1, [&](int i, int j) { return grid_.iv(i, j); });
}

void LFMSimulator::face_set_backward_identity() {
    int nx = grid_.nx, ny = grid_.ny;
    double dx = grid_.dx, dy = grid_.dy;
    auto init = [&](FaceFlowMap2D& f, int axis, int imax, int jmax, auto idx) {
        for (int j = 1; j <= jmax; j++)
            for (int i = 1; i <= imax; i++) {
                int m = idx(i, j);
                double x, y;
                face_coord(axis, i, j, dx, dy, x, y);
                f.bx[m] = x;
                f.by[m] = y;
                f.t0[m] = (axis == 0) ? 1.0 : 0.0;
                f.t1[m] = (axis == 1) ? 1.0 : 0.0;
            }
    };
    init(fmu_, 0, nx - 1, ny, [&](int i, int j) { return grid_.iu(i, j); });
    init(fmv_, 1, nx, ny - 1, [&](int i, int j) { return grid_.iv(i, j); });
}

// RK4 for one face point. State s[4] = [pos(2), covector T(2)]. dp/dt=u(p),
// dT[a]/dt = Σ_c g[2a+c]·T[c] — the matching column of the full 2×2 Jacobian.
void LFMSimulator::face_march_point(int /*axis*/, double& px, double& py, double T[2],
                                    const std::vector<double>& u, const std::vector<double>& v,
                                    double dt_march) const {
    double s0[4] = {px, py, T[0], T[1]};
    auto rhs     = [&](const double s[4], double d[4]) {
        double vu, vv, g[4];
        sample_velocity_gradient(s[0], s[1], u, v, vu, vv, g);
        d[0] = vu;
        d[1] = vv;
        for (int a = 0; a < 2; a++)
            d[2 + a] = g[2 * a + 0] * s[2 + 0] + g[2 * a + 1] * s[2 + 1];
    };
    double k1[4], k2[4], k3[4], k4[4], tmp[4], out[4];
    rhs(s0, k1);
    for (int m = 0; m < 4; m++) {
        k1[m] *= dt_march;
        tmp[m] = s0[m] + 0.5 * k1[m];
    }
    rhs(tmp, k2);
    for (int m = 0; m < 4; m++) {
        k2[m] *= dt_march;
        tmp[m] = s0[m] + 0.5 * k2[m];
    }
    rhs(tmp, k3);
    for (int m = 0; m < 4; m++) {
        k3[m] *= dt_march;
        tmp[m] = s0[m] + k3[m];
    }
    rhs(tmp, k4);
    for (int m = 0; m < 4; m++) {
        k4[m] *= dt_march;
        out[m] = s0[m] + (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) / 6.0;
    }
    px = std::max(0.0, std::min(grid_.Lx(), out[0]));
    py = std::max(0.0, std::min(grid_.Ly(), out[1]));
    for (int a = 0; a < 2; a++) {
        double val = out[2 + a];
        if (!std::isfinite(val))
            val = 0.0;
        T[a] = val;
    }
}

void LFMSimulator::face_march_forward(const std::vector<double>& u, const std::vector<double>& v,
                                      double dt_march) {
    int nx = grid_.nx, ny = grid_.ny;
    auto march = [&](FaceFlowMap2D& f, int axis, int imax, int jmax, auto idx) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= jmax; j++)
            for (int i = 1; i <= imax; i++) {
                int m    = idx(i, j);
                double T[2] = {f.f0[m], f.f1[m]};
                face_march_point(axis, f.fx[m], f.fy[m], T, u, v, dt_march);
                f.f0[m] = T[0];
                f.f1[m] = T[1];
            }
    };
    march(fmu_, 0, nx - 1, ny, [&](int i, int j) { return grid_.iu(i, j); });
    march(fmv_, 1, nx, ny - 1, [&](int i, int j) { return grid_.iv(i, j); });
}

void LFMSimulator::face_march_backward(const std::vector<double>& u, const std::vector<double>& v,
                                       double dt_march) {
    int nx = grid_.nx, ny = grid_.ny;
    auto march = [&](FaceFlowMap2D& f, int axis, int imax, int jmax, auto idx) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= jmax; j++)
            for (int i = 1; i <= imax; i++) {
                int m    = idx(i, j);
                double T[2] = {f.t0[m], f.t1[m]};
                face_march_point(axis, f.bx[m], f.by[m], T, u, v, dt_march);
                f.t0[m] = T[0];
                f.t1[m] = T[1];
            }
    };
    march(fmu_, 0, nx - 1, ny, [&](int i, int j) { return grid_.iu(i, j); });
    march(fmv_, 1, nx, ny - 1, [&](int i, int j) { return grid_.iv(i, j); });
}

// Per-face pullback m_a = T_a · src(ψ_a): sample the staggered src field at the
// (backward, default) face position and project onto the covector row.
void LFMSimulator::face_pullback(const std::vector<double>& su, const std::vector<double>& sv,
                                 std::vector<double>& dst_u, std::vector<double>& dst_v,
                                 bool fwd) const {
    int nx = grid_.nx, ny = grid_.ny;
    auto pull = [&](const FaceFlowMap2D& f, std::vector<double>& dst, int imax, int jmax, auto idx) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= jmax; j++)
            for (int i = 1; i <= imax; i++) {
                int m     = idx(i, j);
                double px = fwd ? f.fx[m] : f.bx[m];
                double py = fwd ? f.fy[m] : f.by[m];
                double T0 = fwd ? f.f0[m] : f.t0[m];
                double T1 = fwd ? f.f1[m] : f.t1[m];
                double sX, sY;
                sample_velocity(px, py, su, sv, sX, sY);
                dst[m] = T0 * sX + T1 * sY;
            }
    };
    pull(fmu_, dst_u, nx - 1, ny, [&](int i, int j) { return grid_.iu(i, j); });
    pull(fmv_, dst_v, nx, ny - 1, [&](int i, int j) { return grid_.iv(i, j); });
}

// Clamp each m component to the [min,max] of its 4 same-face-grid neighbours'
// pre-correction values (author BfeccClamp, per axis grid).
void LFMSimulator::face_bfecc_clamp(const std::vector<double>& pre_u,
                                    const std::vector<double>& pre_v) {
    int nx = grid_.nx, ny = grid_.ny;
    auto clamp = [&](std::vector<double>& m, const std::vector<double>& pre, int imax, int jmax,
                     auto idx) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= jmax; j++)
            for (int i = 1; i <= imax; i++) {
                double lo = 0.0, hi = 0.0;
                bool first    = true;
                auto consider = [&](int a, int b) {
                    if (a < 1 || a > imax || b < 1 || b > jmax)
                        return;
                    double val = pre[idx(a, b)];
                    if (first) {
                        lo = hi = val;
                        first   = false;
                    } else {
                        lo = std::min(lo, val);
                        hi = std::max(hi, val);
                    }
                };
                consider(i - 1, j);
                consider(i + 1, j);
                consider(i, j - 1);
                consider(i, j + 1);
                if (first)
                    continue;
                int id = idx(i, j);
                if (m[id] < lo)
                    m[id] = lo;
                else if (m[id] > hi)
                    m[id] = hi;
            }
    };
    clamp(m_x_, pre_u, nx - 1, ny, [&](int i, int j) { return grid_.iu(i, j); });
    clamp(m_y_, pre_v, nx, ny - 1, [&](int i, int j) { return grid_.iv(i, j); });
}

// Face BFECC error correction (author ReinitAsync sequence, all on faces):
//   m   = T·u0(ψ)                  [already in m_x_/m_y_ from face_pullback]
//   û0  = F·m(φ)                   [forward pullback, sampling staggered m]
//   e   = û0 − u0_face
//   m  -= 0.5 · T·e(ψ)             [error pulled back along the backward map]
//   clamp m to 4 same-face neighbours of the pre-correction m (optional)
// Then writes grid_.u/v ← m directly (NO center↔face gauge averaging).
void LFMSimulator::face_error_correction(const Grid& u0_grid) {
    int nx = grid_.nx, ny = grid_.ny;
    size_t us = (size_t)grid_.u_size(), vs = (size_t)grid_.v_size();

    // û0 = F·m(φ): forward pullback of the just-built impulse onto faces.
    std::vector<double> uh_u(us, 0.0), uh_v(vs, 0.0);
    face_pullback(m_x_, m_y_, uh_u, uh_v, /*fwd=*/true);

    // e = û0 − u0_face (per face component, on the same MAC grid as m).
    std::vector<double> e_u(us, 0.0), e_v(vs, 0.0);
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx - 1; i++) {
            int id  = grid_.iu(i, j);
            e_u[id] = uh_u[id] - u0_grid.u[id];
        }
    for (int j = 1; j <= ny - 1; j++)
        for (int i = 1; i <= nx; i++) {
            int id  = grid_.iv(i, j);
            e_v[id] = uh_v[id] - u0_grid.v[id];
        }

    // corr = T·e(ψ): pull the face error back along the backward map.
    std::vector<double> c_u(us, 0.0), c_v(vs, 0.0);
    face_pullback(e_u, e_v, c_u, c_v, /*fwd=*/false);

    // Pre-correction impulse for the clamp.
    std::vector<double> pre_u, pre_v;
    if (cfg_.lfm_bfecc_clamp) {
        pre_u = m_x_;
        pre_v = m_y_;
    }
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx - 1; i++)
            m_x_[grid_.iu(i, j)] -= 0.5 * c_u[grid_.iu(i, j)];
    for (int j = 1; j <= ny - 1; j++)
        for (int i = 1; i <= nx; i++)
            m_y_[grid_.iv(i, j)] -= 0.5 * c_v[grid_.iv(i, j)];

    if (cfg_.lfm_bfecc_clamp)
        face_bfecc_clamp(pre_u, pre_v);

    // Write the corrected impulse straight into the velocity faces — the impulse
    // IS the velocity (gauge), no averaging. The cycle-end projection follows.
    for (int j = 1; j <= ny; j++)
        for (int i = 1; i <= nx - 1; i++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i + 1, j))
                continue;
            grid_.u_at(i, j) = m_x_[grid_.iu(i, j)];
        }
    for (int j = 1; j <= ny - 1; j++)
        for (int i = 1; i <= nx; i++) {
            if (grid_.is_solid(i, j) || grid_.is_solid(i, j + 1))
                continue;
            grid_.v_at(i, j) = m_y_[grid_.iv(i, j)];
        }
}

// Stubs for test accessors (gauge projection is inlined in run_cycle)
void LFMSimulator::gauge_project() {}
void LFMSimulator::clamp_out_of_solid(double&, double&) const {}

void LFMSimulator::write_frame(int frame_num) {
    mkdir(cfg_.out_dir.c_str(), 0755);
    VtkWriter::write(grid_, frame_num, cfg_);
}

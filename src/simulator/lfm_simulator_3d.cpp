#include "simulator/lfm_simulator_3d.h"
#include "numerics/pressure/pressure_3d.h"
#include <algorithm>
#include <cmath>

// ════════════════════════════════════════════════════════════════════
// 3D LFM simulator — port of src/simulator/lfm_simulator.cpp (Algorithm 1
// of Sun et al. SIGGRAPH 2025) onto the Grid3D / Solver3D stack. The
// structure mirrors the 2D file one-to-one; everything is lifted by adding
// the k-axis, the w velocity component, and the 9-component (3×3) flow-map
// Jacobian in place of the 2D 4-component one.
// ════════════════════════════════════════════════════════════════════

LFMSimulator3D::LFMSimulator3D(const Config& cfg, std::unique_ptr<Solver3D> solver)
    : cfg_(cfg), grid_(cfg.NX, cfg.NY, cfg.NZ, cfg.Lx, cfg.Ly, cfg.Lz), solver_(std::move(solver)),
      bcs_(bc::free_slip_box()),
      flow_map_(cfg.NX, cfg.NY, cfg.NZ, cfg.Lx / cfg.NX, cfg.Ly / cfg.NY, cfg.Lz / cfg.NZ) {
    size_t N = (size_t)cfg_.NX * cfg_.NY * cfg_.NZ;
    for (auto* p : {&m_x_, &m_y_, &m_z_, &phi_mid_x_, &phi_mid_y_, &phi_mid_z_, &F_mid_00_,
                    &F_mid_01_, &F_mid_02_, &F_mid_10_, &F_mid_11_, &F_mid_12_, &F_mid_20_,
                    &F_mid_21_, &F_mid_22_})
        p->assign(N, 0.0);
    apply_bc();
}

void LFMSimulator3D::apply_bc() {
    if (bcs_.empty()) {
        bc::FreeSlipAllFaces3D walls;
        walls.apply(grid_);
        bc::NoSlipImmersedSolid3D solid;
        solid.apply(grid_);
        return;
    }
    bcs_.apply(grid_);
}

void LFMSimulator3D::step() {
    run_cycle(cfg_.lfm_cycle_steps);
}

void LFMSimulator3D::project(Grid3D& g) {
    PressureProjection3D::project(g, cfg_.dt, *solver_, cfg_.solve_iters, cfg_.solve_tol);
}

// ════════════════════════════════════════════════════════════════════
// Algorithm 1: LFM Reinitialization Cycle (3D)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::run_cycle(int n_steps) {
    double dt  = cfg_.dt;
    double rho = 1.0;
    double mu  = (cfg_.Re > 0) ? cfg_.U_inf * 2.0 * cfg_.cyl_R / cfg_.Re : 0.0;
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;

    Grid3D u0_grid = grid_;
    if (mu > 0) {
        std::vector<double> vu, vv, vw;
        compute_viscous(u0_grid, vu, vv, vw);
        accumulate_to_u0(u0_grid, vu, vv, vw, dt / (2.0 * rho)); // Step 1: half-step viscous
    }

    vel_buffer_.clear();
    vel_buffer_.resize(n_steps);
    flow_map_.set_identity();

    // ── Steps 2-5: first midpoint u_{1/2} ──
    Grid3D u_half = grid_;
    rk2_advect(u_half, grid_, grid_.u, grid_.v, grid_.w, dt / 2.0);
    bcs_.apply(u_half);
    project(u_half);
    bcs_.apply(u_half);
    vel_buffer_[0] = {u_half.u, u_half.v, u_half.w};

    save_flow_map_state();
    rk4_march_forward(vel_buffer_[0].u, vel_buffer_[0].v, vel_buffer_[0].w, dt);
    compute_midpoints();
    if (mu > 0) {
        std::vector<double> vu, vv, vw;
        compute_viscous(u_half, vu, vv, vw);
        accumulate_path_integral(u0_grid, vu, vv, vw, dt / rho);
    }

    // ── Steps 6-10: second midpoint u_{3/2} ──
    if (n_steps >= 2) {
        if (mu > 0) {
            std::vector<double> vu, vv, vw;
            compute_viscous(u_half, vu, vv, vw);
            accumulate_to_u0(u_half, vu, vv, vw, dt / rho); // u_{1/2}^† (Step 6)
        }
        Grid3D u_3half = u_half;
        rk2_advect(u_3half, u_half, vel_buffer_[0].u, vel_buffer_[0].v, vel_buffer_[0].w, dt);
        bcs_.apply(u_3half);
        project(u_3half);
        bcs_.apply(u_3half);
        vel_buffer_[1] = {u_3half.u, u_3half.v, u_3half.w};

        save_flow_map_state();
        rk4_march_forward(vel_buffer_[1].u, vel_buffer_[1].v, vel_buffer_[1].w, dt);
        compute_midpoints();
        if (mu > 0) {
            std::vector<double> vu, vv, vw;
            compute_viscous(u_3half, vu, vv, vw);
            accumulate_path_integral(u0_grid, vu, vv, vw, dt / rho);
        }
    }

    // ── Steps 11-17: main leapfrog loop i=2..n-1 ──
    Grid3D u_im32 = grid_;
    Grid3D u_im12 = grid_;
    u_im32.u = vel_buffer_[0].u;
    u_im32.v = vel_buffer_[0].v;
    u_im32.w = vel_buffer_[0].w; // u_{1/2}
    if (n_steps >= 2) {
        u_im12.u = vel_buffer_[1].u;
        u_im12.v = vel_buffer_[1].v;
        u_im12.w = vel_buffer_[1].w; // u_{3/2}
    }
    for (int i_step = 2; i_step < n_steps; i_step++) {
        if (mu > 0) {
            std::vector<double> vu, vv, vw;
            compute_viscous(u_im32, vu, vv, vw);
            accumulate_to_u0(u_im32, vu, vv, vw, 2.0 * dt / rho); // u_{i-3/2}^† (Step 12)
        }
        Grid3D u_next = u_im32;
        rk2_advect(u_next, u_im32, vel_buffer_[i_step - 1].u, vel_buffer_[i_step - 1].v,
                   vel_buffer_[i_step - 1].w, 2.0 * dt);
        bcs_.apply(u_next);
        project(u_next);
        bcs_.apply(u_next);
        vel_buffer_[i_step] = {u_next.u, u_next.v, u_next.w};

        save_flow_map_state();
        rk4_march_forward(vel_buffer_[i_step].u, vel_buffer_[i_step].v, vel_buffer_[i_step].w, dt);
        compute_midpoints();
        if (mu > 0) {
            std::vector<double> vu, vv, vw;
            compute_viscous(u_next, vu, vv, vw);
            accumulate_path_integral(u0_grid, vu, vv, vw, dt / rho);
        }
        u_im32 = u_im12;
        u_im12 = u_next;
    }

    // ── Steps 18-21: backward march ──
    flow_map_.set_backward_identity();
    for (int i_step = n_steps; i_step >= 1; i_step--)
        rk4_march_backward(vel_buffer_[i_step - 1].u, vel_buffer_[i_step - 1].v,
                           vel_buffer_[i_step - 1].w, -dt);

    // ── Step 22: pullback m_n = T^T u_0(Ψ) ──
    pullback_impulse(u0_grid);

    // ── Steps 23-26: error correction ──
    {
        size_t N = (size_t)nx * ny * nz;
        std::vector<double> u_hat_x(N, 0.0), u_hat_y(N, 0.0), u_hat_z(N, 0.0);
        forward_pullback(m_x_, m_y_, m_z_, u_hat_x, u_hat_y, u_hat_z);

        std::vector<double> e_x(N, 0.0), e_y(N, 0.0), e_z(N, 0.0);
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 1; i <= nx; i++) {
                    size_t m   = flow_map_.idx(i, j, k);
                    double uc0 = 0.5 * (u0_grid.u_at(i, j, k) + u0_grid.u_at(i - 1, j, k));
                    double vc0 = 0.5 * (u0_grid.v_at(i, j, k) + u0_grid.v_at(i, j - 1, k));
                    double wc0 = 0.5 * (u0_grid.w_at(i, j, k) + u0_grid.w_at(i, j, k - 1));
                    e_x[m]     = (u_hat_x[m] - uc0) * 0.5;
                    e_y[m]     = (u_hat_y[m] - vc0) * 0.5;
                    e_z[m]     = (u_hat_z[m] - wc0) * 0.5;
                }

        // Un-corrected impulse (paper's `u` before BFECC), kept for the clamp below.
        std::vector<double> m_pre_x, m_pre_y, m_pre_z;
        if (cfg_.lfm_bfecc_clamp) {
            m_pre_x = m_x_;
            m_pre_y = m_y_;
            m_pre_z = m_z_;
        }

        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 1; i <= nx; i++) {
                    size_t m = flow_map_.idx(i, j, k);
                    double X = flow_map_.psi_x[m], Y = flow_map_.psi_y[m], Z = flow_map_.psi_z[m];
                    double ex_s, ey_s, ez_s;
                    sample_cell_centered(e_x, e_y, e_z, X, Y, Z, ex_s, ey_s, ez_s);
                    m_x_[m] -= flow_map_.T00[m] * ex_s + flow_map_.T10[m] * ey_s +
                               flow_map_.T20[m] * ez_s;
                    m_y_[m] -= flow_map_.T01[m] * ex_s + flow_map_.T11[m] * ey_s +
                               flow_map_.T21[m] * ez_s;
                    m_z_[m] -= flow_map_.T02[m] * ex_s + flow_map_.T12[m] * ey_s +
                               flow_map_.T22[m] * ez_s;
                }

        // BFECC clamp (paper's BfeccClampKernel): bound each corrected impulse
        // component to the [min,max] of its 6 face-neighbours' un-corrected
        // values. Prevents the second-order correction from creating new extrema
        // → the cycle stays stable with NO physical viscosity.
        if (cfg_.lfm_bfecc_clamp)
            bfecc_clamp_impulse(m_pre_x, m_pre_y, m_pre_z);
    }

    // ── Step 27: gauge projection u_n ← Project(m_n) ──
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i < nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i + 1, j, k))
                    continue;
                grid_.u_at(i, j, k) =
                    0.5 * (m_x_[flow_map_.idx(i, j, k)] + m_x_[flow_map_.idx(i + 1, j, k)]);
            }
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j < ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i, j + 1, k))
                    continue;
                grid_.v_at(i, j, k) =
                    0.5 * (m_y_[flow_map_.idx(i, j, k)] + m_y_[flow_map_.idx(i, j + 1, k)]);
            }
    for (int k = 1; k < nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i, j, k + 1))
                    continue;
                grid_.w_at(i, j, k) =
                    0.5 * (m_z_[flow_map_.idx(i, j, k)] + m_z_[flow_map_.idx(i, j, k + 1)]);
            }

    double cycle_dt = n_steps * dt;
    PressureProjection3D::project(grid_, cycle_dt, *solver_, cfg_.solve_iters * 2, cfg_.solve_tol);
    bcs_.apply(grid_);

    t_ += n_steps * dt;
    step_++;
}

// ════════════════════════════════════════════════════════════════════
// Midpoint state for path integral (saved before each RK4 step)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::save_flow_map_state() {
    phi_mid_x_ = flow_map_.phi_x;
    phi_mid_y_ = flow_map_.phi_y;
    phi_mid_z_ = flow_map_.phi_z;
    F_mid_00_  = flow_map_.F00;
    F_mid_01_  = flow_map_.F01;
    F_mid_02_  = flow_map_.F02;
    F_mid_10_  = flow_map_.F10;
    F_mid_11_  = flow_map_.F11;
    F_mid_12_  = flow_map_.F12;
    F_mid_20_  = flow_map_.F20;
    F_mid_21_  = flow_map_.F21;
    F_mid_22_  = flow_map_.F22;
}

void LFMSimulator3D::compute_midpoints() {
    size_t N = flow_map_.phi_x.size();
    for (size_t m = 0; m < N; m++) {
        phi_mid_x_[m] = 0.5 * (phi_mid_x_[m] + flow_map_.phi_x[m]);
        phi_mid_y_[m] = 0.5 * (phi_mid_y_[m] + flow_map_.phi_y[m]);
        phi_mid_z_[m] = 0.5 * (phi_mid_z_[m] + flow_map_.phi_z[m]);
        F_mid_00_[m]  = 0.5 * (F_mid_00_[m] + flow_map_.F00[m]);
        F_mid_01_[m]  = 0.5 * (F_mid_01_[m] + flow_map_.F01[m]);
        F_mid_02_[m]  = 0.5 * (F_mid_02_[m] + flow_map_.F02[m]);
        F_mid_10_[m]  = 0.5 * (F_mid_10_[m] + flow_map_.F10[m]);
        F_mid_11_[m]  = 0.5 * (F_mid_11_[m] + flow_map_.F11[m]);
        F_mid_12_[m]  = 0.5 * (F_mid_12_[m] + flow_map_.F12[m]);
        F_mid_20_[m]  = 0.5 * (F_mid_20_[m] + flow_map_.F20[m]);
        F_mid_21_[m]  = 0.5 * (F_mid_21_[m] + flow_map_.F21[m]);
        F_mid_22_[m]  = 0.5 * (F_mid_22_[m] + flow_map_.F22[m]);
    }
}

// ════════════════════════════════════════════════════════════════════
// u_0 += coeff * (cell-centered viscous force averaged to faces).
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::accumulate_to_u0(Grid3D& u0, const std::vector<double>& vu,
                                      const std::vector<double>& vv, const std::vector<double>& vw,
                                      double coeff) {
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    auto& fm = flow_map_;
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i < nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i + 1, j, k))
                    continue;
                u0.u_at(i, j, k) +=
                    coeff * 0.5 * (vu[fm.idx(i, j, k)] + vu[fm.idx(i + 1, j, k)]);
            }
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j < ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i, j + 1, k))
                    continue;
                u0.v_at(i, j, k) +=
                    coeff * 0.5 * (vv[fm.idx(i, j, k)] + vv[fm.idx(i, j + 1, k)]);
            }
    for (int k = 1; k < nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i, j, k + 1))
                    continue;
                u0.w_at(i, j, k) +=
                    coeff * 0.5 * (vw[fm.idx(i, j, k)] + vw[fm.idx(i, j, k + 1)]);
            }
}

// ════════════════════════════════════════════════════════════════════
// Path integral: u_0 += coeff * F_mid^T · visc(Φ_mid).
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::accumulate_path_integral(Grid3D& u0, const std::vector<double>& visc_u,
                                              const std::vector<double>& visc_v,
                                              const std::vector<double>& visc_w, double coeff) {
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    size_t N = (size_t)nx * ny * nz;
    std::vector<double> cx(N, 0.0), cy(N, 0.0), cz(N, 0.0);
    auto& fm = flow_map_;
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k))
                    continue;
                size_t m = fm.idx(i, j, k);
                double vx, vy, vz;
                sample_cell_centered(visc_u, visc_v, visc_w, phi_mid_x_[m], phi_mid_y_[m],
                                     phi_mid_z_[m], vx, vy, vz);
                cx[m] = F_mid_00_[m] * vx + F_mid_10_[m] * vy + F_mid_20_[m] * vz;
                cy[m] = F_mid_01_[m] * vx + F_mid_11_[m] * vy + F_mid_21_[m] * vz;
                cz[m] = F_mid_02_[m] * vx + F_mid_12_[m] * vy + F_mid_22_[m] * vz;
            }
    accumulate_to_u0(u0, cx, cy, cz, coeff);
}

// ════════════════════════════════════════════════════════════════════
// RK2 semi-Lagrangian advection (u/v/w faces)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::rk2_advect(Grid3D& dst, const Grid3D& src, const std::vector<double>& vel_u,
                                const std::vector<double>& vel_v, const std::vector<double>& vel_w,
                                double dt_step) {
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    double dx = grid_.dx, dy = grid_.dy, dz = grid_.dz;

    auto backtrace = [&](double x, double y, double z, double& xo, double& yo, double& zo) {
        double u1, v1, w1;
        sample_velocity(x, y, z, vel_u, vel_v, vel_w, u1, v1, w1);
        double xm = x - 0.5 * dt_step * u1, ym = y - 0.5 * dt_step * v1, zm = z - 0.5 * dt_step * w1;
        double um, vm, wm;
        sample_velocity(xm, ym, zm, vel_u, vel_v, vel_w, um, vm, wm);
        xo = x - dt_step * um;
        yo = y - dt_step * vm;
        zo = z - dt_step * wm;
    };

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i < nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i + 1, j, k)) {
                    dst.u_at(i, j, k) = 0;
                    continue;
                }
                double xs, ys, zs, vu, vv, vw;
                backtrace(i * dx, (j - 0.5) * dy, (k - 0.5) * dz, xs, ys, zs);
                sample_velocity(xs, ys, zs, src.u, src.v, src.w, vu, vv, vw);
                dst.u_at(i, j, k) = vu;
            }
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j < ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i, j + 1, k)) {
                    dst.v_at(i, j, k) = 0;
                    continue;
                }
                double xs, ys, zs, vu, vv, vw;
                backtrace((i - 0.5) * dx, j * dy, (k - 0.5) * dz, xs, ys, zs);
                sample_velocity(xs, ys, zs, src.u, src.v, src.w, vu, vv, vw);
                dst.v_at(i, j, k) = vv;
            }
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k < nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (grid_.is_solid(i, j, k) || grid_.is_solid(i, j, k + 1)) {
                    dst.w_at(i, j, k) = 0;
                    continue;
                }
                double xs, ys, zs, vu, vv, vw;
                backtrace((i - 0.5) * dx, (j - 0.5) * dy, k * dz, xs, ys, zs);
                sample_velocity(xs, ys, zs, src.u, src.v, src.w, vu, vv, vw);
                dst.w_at(i, j, k) = vw;
            }
}

// ════════════════════════════════════════════════════════════════════
// One RK4 step of dΦ/dt = u(Φ), dF/dt = ∇u(Φ)·F for a single cell.
// State s[12]: [0..2] position, [3..11] F row-major (F[3a+b] = ∂Φ_a/∂X_b).
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::march_cell(double& px, double& py, double& pz, double F[9],
                                const std::vector<double>& u, const std::vector<double>& v,
                                const std::vector<double>& w, double dt_march) const {
    double s0[12] = {px, py, pz, F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8]};

    auto rhs = [&](const double s[12], double d[12]) {
        double vu, vv, vw;
        sample_velocity(s[0], s[1], s[2], u, v, w, vu, vv, vw);
        d[0] = vu;
        d[1] = vv;
        d[2] = vw;
        double g[9];
        velocity_gradient_at(s[0], s[1], s[2], u, v, w, g);
        // dF[a][b] = Σ_c (∂u_a/∂x_c) · F[c][b]
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                d[3 + 3 * a + b] = g[3 * a + 0] * s[3 + 0 * 3 + b] + g[3 * a + 1] * s[3 + 1 * 3 + b] +
                                   g[3 * a + 2] * s[3 + 2 * 3 + b];
    };

    double k1[12], k2[12], k3[12], k4[12], tmp[12], out[12];
    rhs(s0, k1);
    for (int m = 0; m < 12; m++) {
        k1[m] *= dt_march;
        tmp[m] = s0[m] + 0.5 * k1[m];
    }
    rhs(tmp, k2);
    for (int m = 0; m < 12; m++) {
        k2[m] *= dt_march;
        tmp[m] = s0[m] + 0.5 * k2[m];
    }
    rhs(tmp, k3);
    for (int m = 0; m < 12; m++) {
        k3[m] *= dt_march;
        tmp[m] = s0[m] + k3[m];
    }
    rhs(tmp, k4);
    for (int m = 0; m < 12; m++) {
        k4[m] *= dt_march;
        out[m] = s0[m] + (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) / 6.0;
    }

    px = std::max(0.0, std::min(grid_.Lx(), out[0]));
    py = std::max(0.0, std::min(grid_.Ly(), out[1]));
    pz = std::max(0.0, std::min(grid_.Lz(), out[2]));
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            double val = out[3 + 3 * a + b];
            if (!std::isfinite(val))
                val = (a == b) ? 1.0 : 0.0;
            F[3 * a + b] = val;
        }
}

void LFMSimulator3D::rk4_march_forward(const std::vector<double>& u, const std::vector<double>& v,
                                       const std::vector<double>& w, double dt_march) {
    int nx = flow_map_.nx, ny = flow_map_.ny, nz = flow_map_.nz;
    auto& fm = flow_map_;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                size_t m = fm.idx(i, j, k);
                double F[9] = {fm.F00[m], fm.F01[m], fm.F02[m], fm.F10[m], fm.F11[m],
                               fm.F12[m], fm.F20[m], fm.F21[m], fm.F22[m]};
                march_cell(fm.phi_x[m], fm.phi_y[m], fm.phi_z[m], F, u, v, w, dt_march);
                fm.F00[m] = F[0];
                fm.F01[m] = F[1];
                fm.F02[m] = F[2];
                fm.F10[m] = F[3];
                fm.F11[m] = F[4];
                fm.F12[m] = F[5];
                fm.F20[m] = F[6];
                fm.F21[m] = F[7];
                fm.F22[m] = F[8];
            }
}

// Backward march: dΨ/dt = u(Ψ), dT/dt = +∇u(Ψ)·T. Combined with dt_march=-dt
// this yields T ≈ F^{-1} (same sign convention fix as the 2D simulator).
void LFMSimulator3D::rk4_march_backward(const std::vector<double>& u, const std::vector<double>& v,
                                        const std::vector<double>& w, double dt_march) {
    int nx = flow_map_.nx, ny = flow_map_.ny, nz = flow_map_.nz;
    auto& fm = flow_map_;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                size_t m = fm.idx(i, j, k);
                double T[9] = {fm.T00[m], fm.T01[m], fm.T02[m], fm.T10[m], fm.T11[m],
                               fm.T12[m], fm.T20[m], fm.T21[m], fm.T22[m]};
                march_cell(fm.psi_x[m], fm.psi_y[m], fm.psi_z[m], T, u, v, w, dt_march);
                fm.T00[m] = T[0];
                fm.T01[m] = T[1];
                fm.T02[m] = T[2];
                fm.T10[m] = T[3];
                fm.T11[m] = T[4];
                fm.T12[m] = T[5];
                fm.T20[m] = T[6];
                fm.T21[m] = T[7];
                fm.T22[m] = T[8];
            }
}

// ════════════════════════════════════════════════════════════════════
// Quadratic B-spline weights (paper §4.1) — same three-point kernel as 2D.
// ════════════════════════════════════════════════════════════════════
static inline void bspline_weights(double r, double w[3]) {
    double a = 0.5 - r;
    double b = 0.5 + r;
    w[0]     = 0.5 * a * a;
    w[1]     = 0.75 - r * r;
    w[2]     = 0.5 * b * b;
}

// ════════════════════════════════════════════════════════════════════
// Velocity interpolation (quadratic B-spline, MAC-grid aware, 27-point)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::sample_velocity(double x, double y, double z, const std::vector<double>& u_vec,
                                     const std::vector<double>& v_vec,
                                     const std::vector<double>& w_vec, double& vu, double& vv,
                                     double& vw) const {
    x = std::max(0.0, std::min(grid_.Lx(), x));
    y = std::max(0.0, std::min(grid_.Ly(), y));
    z = std::max(0.0, std::min(grid_.Lz(), z));
    double dx = grid_.dx, dy = grid_.dy, dz = grid_.dz;
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;

    // Generic 27-point B-spline gather. cx/cy/cz are the (fractional) grid
    // coordinates of the sample point in the target field's index space.
    auto gather = [&](double cx, double cy, double cz, auto at) -> double {
        int ic = (int)std::floor(cx + 0.5);
        int jc = (int)std::floor(cy + 0.5);
        int kc = (int)std::floor(cz + 0.5);
        double wx[3], wy[3], wz[3];
        bspline_weights(cx - ic, wx);
        bspline_weights(cy - jc, wy);
        bspline_weights(cz - kc, wz);
        double s = 0.0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++)
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * at(ic + di, jc + dj, kc + dk);
        return s;
    };

    // u-face value u_vec[iu(i,j,k)] sits at (i·dx, (j-0.5)·dy, (k-0.5)·dz)
    vu = gather(x / dx, y / dy + 0.5, z / dz + 0.5, [&](int ii, int jj, int kk) {
        ii = std::max(0, std::min(nx, ii));
        jj = std::max(1, std::min(ny, jj));
        kk = std::max(1, std::min(nz, kk));
        return u_vec[grid_.iu(ii, jj, kk)];
    });
    // v-face value v_vec[iv(i,j,k)] sits at ((i-0.5)·dx, j·dy, (k-0.5)·dz)
    vv = gather(x / dx + 0.5, y / dy, z / dz + 0.5, [&](int ii, int jj, int kk) {
        ii = std::max(1, std::min(nx, ii));
        jj = std::max(0, std::min(ny, jj));
        kk = std::max(1, std::min(nz, kk));
        return v_vec[grid_.iv(ii, jj, kk)];
    });
    // w-face value w_vec[iw(i,j,k)] sits at ((i-0.5)·dx, (j-0.5)·dy, k·dz)
    vw = gather(x / dx + 0.5, y / dy + 0.5, z / dz, [&](int ii, int jj, int kk) {
        ii = std::max(1, std::min(nx, ii));
        jj = std::max(1, std::min(ny, jj));
        kk = std::max(0, std::min(nz, kk));
        return w_vec[grid_.iw(ii, jj, kk)];
    });
}

// Quadratic B-spline at cell centers ((i-0.5)·dx, (j-0.5)·dy, (k-0.5)·dz).
void LFMSimulator3D::sample_cell_centered(const std::vector<double>& sx,
                                          const std::vector<double>& sy,
                                          const std::vector<double>& sz, double x, double y,
                                          double z, double& vx, double& vy, double& vz) const {
    x = std::max(0.0, std::min(grid_.Lx(), x));
    y = std::max(0.0, std::min(grid_.Ly(), y));
    z = std::max(0.0, std::min(grid_.Lz(), z));
    double dx = grid_.dx, dy = grid_.dy, dz = grid_.dz;
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    double cix = x / dx + 0.5, ciy = y / dy + 0.5, ciz = z / dz + 0.5;
    int ic = (int)std::floor(cix + 0.5);
    int jc = (int)std::floor(ciy + 0.5);
    int kc = (int)std::floor(ciz + 0.5);
    double wx[3], wy[3], wz[3];
    bspline_weights(cix - ic, wx);
    bspline_weights(ciy - jc, wy);
    bspline_weights(ciz - kc, wz);
    vx = vy = vz = 0.0;
    for (int dk = -1; dk <= 1; dk++) {
        int kk = std::max(1, std::min(nz, kc + dk));
        for (int dj = -1; dj <= 1; dj++) {
            int jj = std::max(1, std::min(ny, jc + dj));
            for (int di = -1; di <= 1; di++) {
                int ii   = std::max(1, std::min(nx, ic + di));
                double wgt = wx[di + 1] * wy[dj + 1] * wz[dk + 1];
                size_t m = flow_map_.idx(ii, jj, kk);
                vx += wgt * sx[m];
                vy += wgt * sy[m];
                vz += wgt * sz[m];
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Velocity gradient (3×3) at a cell center. g[3a+b] = ∂u_a/∂x_b.
// Diagonals use the cell's own two faces; off-diagonals use central
// differences across neighbour cells, zeroed when a neighbour is solid.
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::velocity_gradient_at(double x, double y, double z,
                                          const std::vector<double>& ug,
                                          const std::vector<double>& vg,
                                          const std::vector<double>& wg, double g[9]) const {
    int ci = std::max(1, std::min(grid_.nx, (int)(x / grid_.dx + 0.5)));
    int cj = std::max(1, std::min(grid_.ny, (int)(y / grid_.dy + 0.5)));
    int ck = std::max(1, std::min(grid_.nz, (int)(z / grid_.dz + 0.5)));
    velocity_gradient(ci, cj, ck, ug, vg, wg, g);
}

void LFMSimulator3D::velocity_gradient(int i, int j, int k, const std::vector<double>& ug,
                                       const std::vector<double>& vg, const std::vector<double>& wg,
                                       double g[9]) const {
    double dx = grid_.dx, dy = grid_.dy, dz = grid_.dz;
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    int ip1 = std::min(i + 1, nx), im1 = std::max(i - 1, 1);
    int jp1 = std::min(j + 1, ny), jm1 = std::max(j - 1, 1);
    int kp1 = std::min(k + 1, nz), km1 = std::max(k - 1, 1);

    bool sxm = grid_.is_solid(i - 1, j, k), sxp = grid_.is_solid(i + 1, j, k);
    bool sym = grid_.is_solid(i, j - 1, k), syp = grid_.is_solid(i, j + 1, k);
    bool szm = grid_.is_solid(i, j, k - 1), szp = grid_.is_solid(i, j, k + 1);

    auto& m = grid_;
    // Diagonals: cell's own face difference.
    g[0] = (ug[m.iu(i, j, k)] - ug[m.iu(i - 1, j, k)]) / dx; // du/dx
    g[4] = (vg[m.iv(i, j, k)] - vg[m.iv(i, j - 1, k)]) / dy; // dv/dy
    g[8] = (wg[m.iw(i, j, k)] - wg[m.iw(i, j, k - 1)]) / dz; // dw/dz
    // Off-diagonals: central difference of the face value across neighbours.
    g[1] = (sym || syp) ? 0.0 : (ug[m.iu(i, jp1, k)] - ug[m.iu(i, jm1, k)]) / (2 * dy); // du/dy
    g[2] = (szm || szp) ? 0.0 : (ug[m.iu(i, j, kp1)] - ug[m.iu(i, j, km1)]) / (2 * dz); // du/dz
    g[3] = (sxm || sxp) ? 0.0 : (vg[m.iv(ip1, j, k)] - vg[m.iv(im1, j, k)]) / (2 * dx); // dv/dx
    g[5] = (szm || szp) ? 0.0 : (vg[m.iv(i, j, kp1)] - vg[m.iv(i, j, km1)]) / (2 * dz); // dv/dz
    g[6] = (sxm || sxp) ? 0.0 : (wg[m.iw(ip1, j, k)] - wg[m.iw(im1, j, k)]) / (2 * dx); // dw/dx
    g[7] = (sym || syp) ? 0.0 : (wg[m.iw(i, jp1, k)] - wg[m.iw(i, jm1, k)]) / (2 * dy); // dw/dy
}

// ════════════════════════════════════════════════════════════════════
// Viscous force μ∇²u at cell centers (per component)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::compute_viscous(const Grid3D& g, std::vector<double>& vu,
                                     std::vector<double>& vv, std::vector<double>& vw) {
    int nx = g.nx, ny = g.ny, nz = g.nz;
    double mu = (cfg_.Re > 0) ? cfg_.U_inf * 2 * cfg_.cyl_R / cfg_.Re : 0;
    double idx2 = 1.0 / (g.dx * g.dx), idy2 = 1.0 / (g.dy * g.dy), idz2 = 1.0 / (g.dz * g.dz);
    size_t N = (size_t)nx * ny * nz;
    vu.assign(N, 0);
    vv.assign(N, 0);
    vw.assign(N, 0);
    auto& fm = flow_map_;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                if (g.is_solid(i, j, k))
                    continue;
                size_t m = fm.idx(i, j, k);
                // u component (cell-centered = average of the two x-faces)
                double uc = 0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k));
                double uL = (i > 1) ? 0.5 * (g.u_at(i - 1, j, k) + g.u_at(i - 2, j, k)) : uc;
                double uR = (i < nx) ? 0.5 * (g.u_at(i + 1, j, k) + g.u_at(i, j, k)) : uc;
                double uB = (j > 1) ? 0.5 * (g.u_at(i, j - 1, k) + g.u_at(i - 1, j - 1, k)) : uc;
                double uT = (j < ny) ? 0.5 * (g.u_at(i, j + 1, k) + g.u_at(i - 1, j + 1, k)) : uc;
                double uF = (k > 1) ? 0.5 * (g.u_at(i, j, k - 1) + g.u_at(i - 1, j, k - 1)) : uc;
                double uK = (k < nz) ? 0.5 * (g.u_at(i, j, k + 1) + g.u_at(i - 1, j, k + 1)) : uc;
                vu[m] = mu * ((uL + uR - 2 * uc) * idx2 + (uB + uT - 2 * uc) * idy2 +
                              (uF + uK - 2 * uc) * idz2);
                // v component
                double vc = 0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k));
                double vL = (i > 1) ? 0.5 * (g.v_at(i - 1, j, k) + g.v_at(i - 1, j - 1, k)) : vc;
                double vR = (i < nx) ? 0.5 * (g.v_at(i + 1, j, k) + g.v_at(i + 1, j - 1, k)) : vc;
                double vB = (j > 1) ? 0.5 * (g.v_at(i, j - 1, k) + g.v_at(i, j - 2, k)) : vc;
                double vT = (j < ny) ? 0.5 * (g.v_at(i, j + 1, k) + g.v_at(i, j, k)) : vc;
                double vF = (k > 1) ? 0.5 * (g.v_at(i, j, k - 1) + g.v_at(i, j - 1, k - 1)) : vc;
                double vK = (k < nz) ? 0.5 * (g.v_at(i, j, k + 1) + g.v_at(i, j - 1, k + 1)) : vc;
                vv[m] = mu * ((vL + vR - 2 * vc) * idx2 + (vB + vT - 2 * vc) * idy2 +
                              (vF + vK - 2 * vc) * idz2);
                // w component
                double wc = 0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1));
                double wL = (i > 1) ? 0.5 * (g.w_at(i - 1, j, k) + g.w_at(i - 1, j, k - 1)) : wc;
                double wR = (i < nx) ? 0.5 * (g.w_at(i + 1, j, k) + g.w_at(i + 1, j, k - 1)) : wc;
                double wB = (j > 1) ? 0.5 * (g.w_at(i, j - 1, k) + g.w_at(i, j - 1, k - 1)) : wc;
                double wT = (j < ny) ? 0.5 * (g.w_at(i, j + 1, k) + g.w_at(i, j + 1, k - 1)) : wc;
                double wF = (k > 1) ? 0.5 * (g.w_at(i, j, k - 1) + g.w_at(i, j, k - 2)) : wc;
                double wK = (k < nz) ? 0.5 * (g.w_at(i, j, k + 1) + g.w_at(i, j, k)) : wc;
                vw[m] = mu * ((wL + wR - 2 * wc) * idx2 + (wB + wT - 2 * wc) * idy2 +
                              (wF + wK - 2 * wc) * idz2);
            }
}

// ════════════════════════════════════════════════════════════════════
// Pullback: m = T^T · u_0(Ψ)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::pullback_impulse(const Grid3D& u0_grid) {
    int nx = flow_map_.nx, ny = flow_map_.ny, nz = flow_map_.nz;
    auto& fm = flow_map_;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                size_t m = fm.idx(i, j, k);
                double uX, uY, uZ;
                sample_velocity(fm.psi_x[m], fm.psi_y[m], fm.psi_z[m], u0_grid.u, u0_grid.v,
                                u0_grid.w, uX, uY, uZ);
                m_x_[m] = fm.T00[m] * uX + fm.T10[m] * uY + fm.T20[m] * uZ;
                m_y_[m] = fm.T01[m] * uX + fm.T11[m] * uY + fm.T21[m] * uZ;
                m_z_[m] = fm.T02[m] * uX + fm.T12[m] * uY + fm.T22[m] * uZ;
            }
}

// ════════════════════════════════════════════════════════════════════
// Forward pullback: û_0 = F^T · m(Φ)
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::forward_pullback(const std::vector<double>& mx, const std::vector<double>& my,
                                      const std::vector<double>& mz, std::vector<double>& ux,
                                      std::vector<double>& uy, std::vector<double>& uz) {
    int nx = flow_map_.nx, ny = flow_map_.ny, nz = flow_map_.nz;
    auto& fm = flow_map_;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                size_t m = fm.idx(i, j, k);
                double msx, msy, msz;
                sample_cell_centered(mx, my, mz, fm.phi_x[m], fm.phi_y[m], fm.phi_z[m], msx, msy,
                                     msz);
                ux[m] = fm.F00[m] * msx + fm.F10[m] * msy + fm.F20[m] * msz;
                uy[m] = fm.F01[m] * msx + fm.F11[m] * msy + fm.F21[m] * msz;
                uz[m] = fm.F02[m] * msx + fm.F12[m] * msy + fm.F22[m] * msz;
            }
}

// ════════════════════════════════════════════════════════════════════
// BFECC clamp (paper's BfeccClampKernel): bound each corrected impulse
// component to the [min,max] of its 6 face-neighbours' un-corrected values.
// Reads `pre` (saved pre-correction copy) and clamps m_{x,y,z}_ in place;
// only the 6 face neighbours are considered (center excluded), matching the
// reference. This is what lets a cycle stay stable with zero viscosity.
// ════════════════════════════════════════════════════════════════════
void LFMSimulator3D::bfecc_clamp_impulse(const std::vector<double>& pre_x,
                                         const std::vector<double>& pre_y,
                                         const std::vector<double>& pre_z) {
    int nx = flow_map_.nx, ny = flow_map_.ny, nz = flow_map_.nz;
    auto& fm = flow_map_;
    auto clamp_field = [&](std::vector<double>& m, const std::vector<double>& pre) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 1; i <= nx; i++) {
                    double lo = 0.0, hi = 0.0;
                    bool first    = true;
                    auto consider = [&](int a, int b, int c) {
                        if (a < 1 || a > nx || b < 1 || b > ny || c < 1 || c > nz)
                            return;
                        double v = pre[fm.idx(a, b, c)];
                        if (first) {
                            lo = hi = v;
                            first   = false;
                        } else {
                            lo = std::min(lo, v);
                            hi = std::max(hi, v);
                        }
                    };
                    consider(i - 1, j, k);
                    consider(i + 1, j, k);
                    consider(i, j - 1, k);
                    consider(i, j + 1, k);
                    consider(i, j, k - 1);
                    consider(i, j, k + 1);
                    if (first)
                        continue;
                    size_t id = fm.idx(i, j, k);
                    if (m[id] < lo)
                        m[id] = lo;
                    else if (m[id] > hi)
                        m[id] = hi;
                }
    };
    clamp_field(m_x_, pre_x);
    clamp_field(m_y_, pre_y);
    clamp_field(m_z_, pre_z);
}

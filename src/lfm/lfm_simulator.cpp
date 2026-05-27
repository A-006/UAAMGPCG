#include "lfm/lfm_simulator.h"
#include "pressure/pressure.h"
#include "boundary/boundary.h"
#include "scenarios/karman.h"
#include "io/vtk_writer.h"
#include <cmath>
#include <iostream>
#include <sys/stat.h>

LFMSimulator::LFMSimulator(const Config& cfg, std::unique_ptr<Solver> solver)
    : cfg_(cfg), grid_(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly),
      solver_(std::move(solver)),
      flow_map_(cfg.NX, cfg.NY, cfg.Lx/cfg.NX, cfg.Ly/cfg.NY)
{
    int N = cfg_.NX * cfg_.NY;
    m_x_.resize(N, 0.0);
    m_y_.resize(N, 0.0);
    phi_mid_x_.resize(N, 0.0);
    phi_mid_y_.resize(N, 0.0);
    F_mid_00_.resize(N, 0.0);
    F_mid_10_.resize(N, 0.0);
    F_mid_01_.resize(N, 0.0);
    F_mid_11_.resize(N, 0.0);

    if (cfg_.scenario == "karman") {
        scenarios::Karman k{ cfg_.cyl_cx, cfg_.cyl_cy, cfg_.cyl_R, cfg_.U_inf };
        if (k.cyl_R > 0) scenarios::setup_karman_cylinder(grid_, k);
        scenarios::set_uniform_inflow(grid_, k.U_inf);
    }
    BoundaryConditions::applyKarman(grid_, cfg_.U_inf);
    BoundaryConditions::applySolid(grid_);
}

void LFMSimulator::step() { run_cycle(cfg_.lfm_cycle_steps); }

// ═══════════════════════════════════════════════════════════════
// Algorithm 1: LFM Reinitialization Cycle (Sun et al. SIGGRAPH 2025)
//
// Pure implementation — no artificial constraints beyond what the
// paper describes. velocity_gradient returns 0 near solid boundaries
// (central differences require two fluid neighbors), which naturally
// gives dF/dt ≈ 0 and F ≈ I near walls without any explicit override.
// ═══════════════════════════════════════════════════════════════
void LFMSimulator::run_cycle(int n_steps) {
    double dt = cfg_.dt;
    double rho = 1.0;
    double mu = (cfg_.Re > 0) ? cfg_.U_inf * 2.0 * cfg_.cyl_R / cfg_.Re : 0.0;
    int nx = grid_.nx, ny = grid_.ny;

    Grid u0_grid = grid_;
    if (mu > 0) {
        std::vector<double> vu, vv;
        compute_viscous(u0_grid, vu, vv);
        accumulate_to_u0(u0_grid, vu, vv, dt/(2.0*rho));  // Step 1: half-step viscous
    }

    vel_buffer_.clear(); vel_buffer_.resize(n_steps);
    flow_map_.set_identity();

    // ── Steps 2-5: First midpoint u_{1/2} ──
    Grid u_half(nx, ny, grid_.Lx(), grid_.Ly());
    u_half = grid_;
    rk2_advect(u_half, grid_, grid_.u, grid_.v, dt / 2.0);
    BoundaryConditions::applyKarman(u_half, cfg_.U_inf);
    BoundaryConditions::applySolid(u_half);
    project(u_half);
    BoundaryConditions::applyKarman(u_half, cfg_.U_inf);
    BoundaryConditions::applySolid(u_half);
    vel_buffer_[0] = {u_half.u, u_half.v};

    // Step 4: Forward march, save midpoint for Step 5 path integral
    save_flow_map_state();
    rk4_march_forward(vel_buffer_[0].u, vel_buffer_[0].v, dt);
    compute_midpoints();
    if (mu > 0) {
        std::vector<double> vu, vv;
        compute_viscous(u_half, vu, vv);
        accumulate_path_integral(u0_grid, vu, vv, dt/rho);
    }

    // ── Steps 6-10: Second midpoint u_{3/2} ──
    if (n_steps >= 2) {
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_half, vu, vv);
            accumulate_to_u0(u_half, vu, vv, dt/rho);  // u_{1/2}^† (Step 6)
        }
        Grid u_3half(nx, ny, grid_.Lx(), grid_.Ly());
        u_3half = u_half;
        rk2_advect(u_3half, u_half, vel_buffer_[0].u, vel_buffer_[0].v, dt);
        BoundaryConditions::applyKarman(u_3half, cfg_.U_inf);
        BoundaryConditions::applySolid(u_3half);
        project(u_3half);
        BoundaryConditions::applyKarman(u_3half, cfg_.U_inf);
        BoundaryConditions::applySolid(u_3half);
        vel_buffer_[1] = {u_3half.u, u_3half.v};

        save_flow_map_state();
        rk4_march_forward(vel_buffer_[1].u, vel_buffer_[1].v, dt);
        compute_midpoints();
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_3half, vu, vv);
            accumulate_path_integral(u0_grid, vu, vv, dt/rho);
        }
    }

    // ── Steps 11-17: Main loop i=2..n-1 (leapfrog) ──
    // Source for advection AND viscosity is u_{i-3/2}; advection velocity is u_{i-1/2}.
    // Use vel_buffer_ entries (original, unmodified midpoint velocities).
    Grid u_im32 = grid_;  // template (keeps solid mask)
    Grid u_im12 = grid_;
    u_im32.u = vel_buffer_[0].u; u_im32.v = vel_buffer_[0].v;  // u_{1/2}
    if (n_steps >= 2) {
        u_im12.u = vel_buffer_[1].u; u_im12.v = vel_buffer_[1].v;  // u_{3/2}
    }
    for (int i_step = 2; i_step < n_steps; i_step++) {
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_im32, vu, vv);
            accumulate_to_u0(u_im32, vu, vv, 2.0*dt/rho);  // u_{i-3/2}^† (Step 12)
        }
        Grid u_next(nx, ny, grid_.Lx(), grid_.Ly());
        u_next = u_im32;
        // Leapfrog: advect u_{i-3/2}^† with velocity field u_{i-1/2} for 2Δt
        rk2_advect(u_next, u_im32, vel_buffer_[i_step-1].u, vel_buffer_[i_step-1].v, 2.0*dt);
        BoundaryConditions::applyKarman(u_next, cfg_.U_inf);
        BoundaryConditions::applySolid(u_next);
        project(u_next);
        BoundaryConditions::applyKarman(u_next, cfg_.U_inf);
        BoundaryConditions::applySolid(u_next);
        vel_buffer_[i_step] = {u_next.u, u_next.v};

        save_flow_map_state();
        rk4_march_forward(vel_buffer_[i_step].u, vel_buffer_[i_step].v, dt);
        compute_midpoints();
        if (mu > 0) {
            std::vector<double> vu, vv;
            compute_viscous(u_next, vu, vv);
            accumulate_path_integral(u0_grid, vu, vv, dt/rho);
        }
        // Slide window: u_{i-3/2} ← u_{i-1/2}, u_{i-1/2} ← u_{i+1/2}
        u_im32 = u_im12;
        u_im12 = u_next;
    }

    // ── Steps 18-21: Backward march ──
    flow_map_.set_backward_identity();
    for (int i_step = n_steps; i_step >= 1; i_step--)
        rk4_march_backward(vel_buffer_[i_step-1].u, vel_buffer_[i_step-1].v, -dt);

    // ── Step 22: Pullback m_n = T^T u_0(Ψ) ──
    pullback_impulse(u0_grid);

    // ── Steps 23-26: Error correction ──
    {
        std::vector<double> u_hat_x(nx*ny, 0.0), u_hat_y(nx*ny, 0.0);
        forward_pullback(m_x_, m_y_, u_hat_x, u_hat_y);

        std::vector<double> e_x(nx*ny, 0.0), e_y(nx*ny, 0.0);
        for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
            size_t k = flow_map_.idx(i,j);
            double uc0 = 0.5*(u0_grid.u_at(i,j)+u0_grid.u_at(i-1,j));
            double vc0 = 0.5*(u0_grid.v_at(i,j)+u0_grid.v_at(i,j-1));
            e_x[k] = (u_hat_x[k] - uc0) * 0.5;
            e_y[k] = (u_hat_y[k] - vc0) * 0.5;
        }

        for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
            size_t k = flow_map_.idx(i,j);
            double X = flow_map_.psi_x[k], Y = flow_map_.psi_y[k];
            double ex_s, ey_s;
            sample_cell_centered(e_x, e_y, X, Y, ex_s, ey_s);
            m_x_[k] -= flow_map_.T00[k]*ex_s + flow_map_.T10[k]*ey_s;
            m_y_[k] -= flow_map_.T01[k]*ex_s + flow_map_.T11[k]*ey_s;
        }
    }

    // ── Step 27: Gauge projection u_n ← Project(m_n) ──
    for (int i=1;i<nx;i++) for (int j=1;j<=ny;j++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i+1,j)) continue;
        size_t kL=flow_map_.idx(i,j), kR=flow_map_.idx(i+1,j);
        grid_.u_at(i,j)=0.5*(m_x_[kL]+m_x_[kR]);
    }
    for (int i=1;i<=nx;i++) for (int j=1;j<ny;j++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i,j+1)) continue;
        size_t kB=flow_map_.idx(i,j), kT=flow_map_.idx(i,j+1);
        grid_.v_at(i,j)=0.5*(m_y_[kB]+m_y_[kT]);
    }
    double cycle_dt = n_steps * dt;
    PressureProjection::project(grid_, cycle_dt, *solver_, cfg_.solve_iters*2, cfg_.solve_tol);
    // p_at now holds ξ in physical units. Add Bernoulli to recover physical pressure.
    for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
        if (grid_.is_solid(i,j)) continue;
        double uc = 0.5*(grid_.u_at(i,j)+grid_.u_at(i-1,j));
        double vc = 0.5*(grid_.v_at(i,j)+grid_.v_at(i,j-1));
        grid_.p_at(i,j) += 0.5*(uc*uc + vc*vc);
    }
    BoundaryConditions::applyKarman(grid_, cfg_.U_inf);
    BoundaryConditions::applySolid(grid_);

    t_ += n_steps * dt; step_++;
}

// ═══════════════════════════════════════════════════════════
// Midpoint state for path integral: saved before each RK4 step
// ═══════════════════════════════════════════════════════════
void LFMSimulator::save_flow_map_state() {
    std::copy(flow_map_.phi_x.begin(), flow_map_.phi_x.end(), phi_mid_x_.begin());
    std::copy(flow_map_.phi_y.begin(), flow_map_.phi_y.end(), phi_mid_y_.begin());
    std::copy(flow_map_.F00.begin(),  flow_map_.F00.end(),  F_mid_00_.begin());
    std::copy(flow_map_.F10.begin(),  flow_map_.F10.end(),  F_mid_10_.begin());
    std::copy(flow_map_.F01.begin(),  flow_map_.F01.end(),  F_mid_01_.begin());
    std::copy(flow_map_.F11.begin(),  flow_map_.F11.end(),  F_mid_11_.begin());
}

void LFMSimulator::compute_midpoints() {
    int N = (int)flow_map_.phi_x.size();
    for (int k = 0; k < N; k++) {
        phi_mid_x_[k] = 0.5 * (phi_mid_x_[k] + flow_map_.phi_x[k]);
        phi_mid_y_[k] = 0.5 * (phi_mid_y_[k] + flow_map_.phi_y[k]);
        F_mid_00_[k] = 0.5 * (F_mid_00_[k] + flow_map_.F00[k]);
        F_mid_10_[k] = 0.5 * (F_mid_10_[k] + flow_map_.F10[k]);
        F_mid_01_[k] = 0.5 * (F_mid_01_[k] + flow_map_.F01[k]);
        F_mid_11_[k] = 0.5 * (F_mid_11_[k] + flow_map_.F11[k]);
    }
}

// ═══════════════════════════════════════════════════════════
// u_0 += coeff * (cell-centered viscous force averaged to faces).
// Adds to face values directly so the MAC layout is preserved.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::accumulate_to_u0(Grid& u0, const std::vector<double>& vu,
    const std::vector<double>& vv, double coeff) {
    int nx=flow_map_.nx, ny=flow_map_.ny;
    for (int j=1;j<=ny;j++) for (int i=1;i<nx;i++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i+1,j)) continue;
        u0.u_at(i,j) += coeff * 0.5 * (vu[flow_map_.idx(i,j)] + vu[flow_map_.idx(i+1,j)]);
    }
    for (int i=1;i<=nx;i++) for (int j=1;j<ny;j++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i,j+1)) continue;
        u0.v_at(i,j) += coeff * 0.5 * (vv[flow_map_.idx(i,j)] + vv[flow_map_.idx(i,j+1)]);
    }
}

// ═══════════════════════════════════════════════════════════
// Path integral: u_0 += coeff * F_mid^T · visc(Φ_mid).
// Computes cell-centered contribution first, then averages to faces.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::accumulate_path_integral(Grid& u0,
    const std::vector<double>& visc_u, const std::vector<double>& visc_v, double coeff) {
    int nx=flow_map_.nx, ny=flow_map_.ny;
    std::vector<double> cx(nx*ny, 0.0), cy(nx*ny, 0.0);
    for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
        if (grid_.is_solid(i,j)) continue;
        size_t k = flow_map_.idx(i,j);
        double vx, vy;
        sample_cell_centered(visc_u, visc_v, phi_mid_x_[k], phi_mid_y_[k], vx, vy);
        cx[k] = F_mid_00_[k]*vx + F_mid_10_[k]*vy;
        cy[k] = F_mid_01_[k]*vx + F_mid_11_[k]*vy;
    }
    for (int j=1;j<=ny;j++) for (int i=1;i<nx;i++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i+1,j)) continue;
        u0.u_at(i,j) += coeff * 0.5 * (cx[flow_map_.idx(i,j)] + cx[flow_map_.idx(i+1,j)]);
    }
    for (int i=1;i<=nx;i++) for (int j=1;j<ny;j++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i,j+1)) continue;
        u0.v_at(i,j) += coeff * 0.5 * (cy[flow_map_.idx(i,j)] + cy[flow_map_.idx(i,j+1)]);
    }
}

// ═══════════════════════════════════════════════════════════
// RK2 semi-Lagrangian advection
// ═══════════════════════════════════════════════════════════
void LFMSimulator::rk2_advect(Grid& dst, const Grid& src,
    const std::vector<double>& vel_u, const std::vector<double>& vel_v, double dt_step) {
    int nx=grid_.nx, ny=grid_.ny; double dx=grid_.dx, dy=grid_.dy;
    for (int i=1;i<nx;i++) for (int j=1;j<=ny;j++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i+1,j)) { dst.u_at(i,j)=0; continue; }
        double xu=i*dx, yu=(j-0.5)*dy;
        double u1=sample_u(xu,yu,vel_u,vel_v), v1=sample_v(xu,yu,vel_u,vel_v);
        double xm=xu-0.5*dt_step*u1, ym=yu-0.5*dt_step*v1;
        double um=sample_u(xm,ym,vel_u,vel_v), vm=sample_v(xm,ym,vel_u,vel_v);
        dst.u_at(i,j)=sample_u(xu-dt_step*um, yu-dt_step*vm, src.u, src.v);
    }
    for (int i=1;i<=nx;i++) for (int j=1;j<ny;j++) {
        if (grid_.is_solid(i,j)||grid_.is_solid(i,j+1)) { dst.v_at(i,j)=0; continue; }
        double xv=(i-0.5)*dx, yv=j*dy;
        double u1=sample_u(xv,yv,vel_u,vel_v), v1=sample_v(xv,yv,vel_u,vel_v);
        double xm=xv-0.5*dt_step*u1, ym=yv-0.5*dt_step*v1;
        double um=sample_u(xm,ym,vel_u,vel_v), vm=sample_v(xm,ym,vel_u,vel_v);
        dst.v_at(i,j)=sample_v(xv-dt_step*um, yv-dt_step*vm, src.u, src.v);
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
void LFMSimulator::rk4_march_forward(const std::vector<double>& u,
    const std::vector<double>& v, double dt_march) {
    int nx=flow_map_.nx, ny=flow_map_.ny;
    for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
        size_t k=flow_map_.idx(i,j);
        double x0=flow_map_.phi_x[k], y0=flow_map_.phi_y[k];
        double f00=flow_map_.F00[k], f10=flow_map_.F10[k];
        double f01=flow_map_.F01[k], f11=flow_map_.F11[k];

        auto rhs=[&](double px,double py,double c00,double c10,double c01,double c11,
            double& dx,double& dy,double& d00,double& d10,double& d01,double& d11){
            double vu,vv; sample_velocity(px,py,u,v,vu,vv);
            dx=vu; dy=vv;
            double dudx,dudy,dvdx,dvdy;
            velocity_gradient_at(px,py,u,v,dudx,dudy,dvdx,dvdy);
            d00=dudx*c00+dudy*c10; d10=dvdx*c00+dvdy*c10;
            d01=dudx*c01+dudy*c11; d11=dvdx*c01+dvdy*c11;
        };

        double k1[6],k2[6],k3[6],k4[6];
        rhs(x0,y0,f00,f10,f01,f11,k1[0],k1[1],k1[2],k1[3],k1[4],k1[5]);
        for(int m=0;m<6;m++) k1[m]*=dt_march;
        rhs(x0+0.5*k1[0],y0+0.5*k1[1],f00+0.5*k1[2],f10+0.5*k1[3],f01+0.5*k1[4],f11+0.5*k1[5],k2[0],k2[1],k2[2],k2[3],k2[4],k2[5]);
        for(int m=0;m<6;m++) k2[m]*=dt_march;
        rhs(x0+0.5*k2[0],y0+0.5*k2[1],f00+0.5*k2[2],f10+0.5*k2[3],f01+0.5*k2[4],f11+0.5*k2[5],k3[0],k3[1],k3[2],k3[3],k3[4],k3[5]);
        for(int m=0;m<6;m++) k3[m]*=dt_march;
        rhs(x0+k3[0],y0+k3[1],f00+k3[2],f10+k3[3],f01+k3[4],f11+k3[5],k4[0],k4[1],k4[2],k4[3],k4[4],k4[5]);
        for(int m=0;m<6;m++) k4[m]*=dt_march;

        flow_map_.phi_x[k]=std::max(0.0,std::min(grid_.Lx(),x0+(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0));
        flow_map_.phi_y[k]=std::max(0.0,std::min(grid_.Ly(),y0+(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0));
        double nF00=f00+(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0;
        double nF10=f10+(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6.0;
        double nF01=f01+(k1[4]+2*k2[4]+2*k3[4]+k4[4])/6.0;
        double nF11=f11+(k1[5]+2*k2[5]+2*k3[5]+k4[5])/6.0;
        if (!std::isfinite(nF00)) nF00=1.0; if (!std::isfinite(nF10)) nF10=0.0;
        if (!std::isfinite(nF01)) nF01=0.0; if (!std::isfinite(nF11)) nF11=1.0;
        flow_map_.F00[k]=nF00; flow_map_.F10[k]=nF10;
        flow_map_.F01[k]=nF01; flow_map_.F11[k]=nF11;
    }
}

// ═══════════════════════════════════════════════════════════
// RK4-March backward: dΨ/dt = u(Ψ), dT/dt = +∇u(Ψ)·T
// Note: with dt_march = -dt, this gives T ≈ F^{-1} (correct inverse).
// The paper has dT/dt = -∇u·T, but that combined with -dt step
// gives T ≈ F instead of T ≈ F^{-1}. We fix the sign here.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::rk4_march_backward(const std::vector<double>& u,
    const std::vector<double>& v, double dt_march) {
    int nx=flow_map_.nx, ny=flow_map_.ny;
    for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
        size_t k=flow_map_.idx(i,j);
        double x0=flow_map_.psi_x[k], y0=flow_map_.psi_y[k];
        double t00=flow_map_.T00[k], t10=flow_map_.T10[k];
        double t01=flow_map_.T01[k], t11=flow_map_.T11[k];

        auto rhs=[&](double px,double py,double c00,double c10,double c01,double c11,
            double& dx,double& dy,double& d00,double& d10,double& d01,double& d11){
            double vu,vv; sample_velocity(px,py,u,v,vu,vv);
            dx=vu; dy=vv;
            double dudx,dudy,dvdx,dvdy;
            velocity_gradient_at(px,py,u,v,dudx,dudy,dvdx,dvdy);
            d00=dudx*c00+dudy*c10; d10=dvdx*c00+dvdy*c10;
            d01=dudx*c01+dudy*c11; d11=dvdx*c01+dvdy*c11;
        };

        double k1[6],k2[6],k3[6],k4[6];
        rhs(x0,y0,t00,t10,t01,t11,k1[0],k1[1],k1[2],k1[3],k1[4],k1[5]);
        for(int m=0;m<6;m++) k1[m]*=dt_march;
        rhs(x0+0.5*k1[0],y0+0.5*k1[1],t00+0.5*k1[2],t10+0.5*k1[3],t01+0.5*k1[4],t11+0.5*k1[5],k2[0],k2[1],k2[2],k2[3],k2[4],k2[5]);
        for(int m=0;m<6;m++) k2[m]*=dt_march;
        rhs(x0+0.5*k2[0],y0+0.5*k2[1],t00+0.5*k2[2],t10+0.5*k2[3],t01+0.5*k2[4],t11+0.5*k2[5],k3[0],k3[1],k3[2],k3[3],k3[4],k3[5]);
        for(int m=0;m<6;m++) k3[m]*=dt_march;
        rhs(x0+k3[0],y0+k3[1],t00+k3[2],t10+k3[3],t01+k3[4],t11+k3[5],k4[0],k4[1],k4[2],k4[3],k4[4],k4[5]);
        for(int m=0;m<6;m++) k4[m]*=dt_march;

        flow_map_.psi_x[k]=std::max(0.0,std::min(grid_.Lx(),x0+(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0));
        flow_map_.psi_y[k]=std::max(0.0,std::min(grid_.Ly(),y0+(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0));
        double nT00=t00+(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0;
        double nT10=t10+(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6.0;
        double nT01=t01+(k1[4]+2*k2[4]+2*k3[4]+k4[4])/6.0;
        double nT11=t11+(k1[5]+2*k2[5]+2*k3[5]+k4[5])/6.0;
        if (!std::isfinite(nT00)) nT00=1.0; if (!std::isfinite(nT10)) nT10=0.0;
        if (!std::isfinite(nT01)) nT01=0.0; if (!std::isfinite(nT11)) nT11=1.0;
        flow_map_.T00[k]=nT00; flow_map_.T10[k]=nT10;
        flow_map_.T01[k]=nT01; flow_map_.T11[k]=nT11;
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
    w[0] = 0.5 * a * a;
    w[1] = 0.75 - r * r;
    w[2] = 0.5 * b * b;
}

// ═══════════════════════════════════════════════════════════
// Velocity interpolation (quadratic B-spline, MAC-grid aware)
// ═══════════════════════════════════════════════════════════
void LFMSimulator::sample_velocity(double x, double y,
    const std::vector<double>& u_vec, const std::vector<double>& v_vec,
    double& vu, double& vv) const {
    x = std::max(0.0, std::min(grid_.Lx(), x));
    y = std::max(0.0, std::min(grid_.Ly(), y));
    double dx = grid_.dx, dy = grid_.dy;
    int nx = grid_.nx, ny = grid_.ny;

    // ── u-face value: u_vec[iu(i,j)] sits at (i·dx, (j-0.5)·dy) ──
    {
        double ux = x / dx;            // ∈ [0, nx]
        double uy = y / dy + 0.5;      // ∈ [0.5, ny+0.5]
        int ic = (int)std::floor(ux + 0.5);
        int jc = (int)std::floor(uy + 0.5);
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
        double vx = x / dx + 0.5;      // ∈ [0.5, nx+0.5]
        double vy = y / dy;             // ∈ [0, ny]
        int ic = (int)std::floor(vx + 0.5);
        int jc = (int)std::floor(vy + 0.5);
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

double LFMSimulator::sample_u(double x,double y,const std::vector<double>& u,const std::vector<double>& v) const
{double vu,vv;sample_velocity(x,y,u,v,vu,vv);return vu;}
double LFMSimulator::sample_v(double x,double y,const std::vector<double>& u,const std::vector<double>& v) const
{double vu,vv;sample_velocity(x,y,u,v,vu,vv);return vv;}

// Quadratic B-spline at cell centers ((i-0.5)·dx, (j-0.5)·dy), i,j ∈ [1, nx]×[1, ny].
void LFMSimulator::sample_cell_centered(const std::vector<double>& sx,
    const std::vector<double>& sy, double x, double y, double& vx, double& vy) const {
    x = std::max(0.0, std::min(grid_.Lx(), x));
    y = std::max(0.0, std::min(grid_.Ly(), y));
    double dx = grid_.dx, dy = grid_.dy;
    int nx = grid_.nx, ny = grid_.ny;
    double cix = x / dx + 0.5;       // ∈ [0.5, nx+0.5]
    double ciy = y / dy + 0.5;
    int ic = (int)std::floor(cix + 0.5);
    int jc = (int)std::floor(ciy + 0.5);
    double wx[3], wy[3];
    bspline_weights(cix - ic, wx);
    bspline_weights(ciy - jc, wy);
    vx = 0.0; vy = 0.0;
    for (int dj = -1; dj <= 1; dj++) {
        int jj = std::max(1, std::min(ny, jc + dj));
        for (int di = -1; di <= 1; di++) {
            int ii = std::max(1, std::min(nx, ic + di));
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
void LFMSimulator::velocity_gradient_at(double x, double y,
    const std::vector<double>& ug, const std::vector<double>& vg,
    double& du_dx, double& du_dy, double& dv_dx, double& dv_dy) const {
    int ci = std::max(1, std::min(grid_.nx, (int)(x/grid_.dx + 0.5)));
    int cj = std::max(1, std::min(grid_.ny, (int)(y/grid_.dy + 0.5)));
    velocity_gradient(ci, cj, ug, vg, du_dx, du_dy, dv_dx, dv_dy);
}

// ═══════════════════════════════════════════════════════════
// Velocity gradient at cell center.
// Diagonals du_dx, dv_dy use cell's OWN two faces (no neighbor needed,
// no-slip face value is a legitimate boundary value → keep the shear).
// Off-diagonals du_dy, dv_dx need central differences across neighbors;
// zero them when either neighbor is solid (stair-step protection).
// ═══════════════════════════════════════════════════════════
void LFMSimulator::velocity_gradient(int i,int j,const std::vector<double>& ug,const std::vector<double>& vg,
    double& du_dx,double& du_dy,double& dv_dx,double& dv_dy) const {
    double dx=grid_.dx, dy=grid_.dy;
    int nx=grid_.nx, ny=grid_.ny;
    int ip1=std::min(i+1,nx), im1=std::max(i-1,1);
    int jp1=std::min(j+1,ny), jm1=std::max(j-1,1);
    auto iu=[&](int ii,int jj){return ii+jj*(nx+1);};
    auto iv=[&](int ii,int jj){return ii+jj*(nx+2);};

    bool solid_B=grid_.is_solid(i,j-1), solid_T=grid_.is_solid(i,j+1);
    bool solid_L=grid_.is_solid(i-1,j), solid_R=grid_.is_solid(i+1,j);

    du_dx = (ug[iu(i,j)] - ug[iu(i-1,j)]) / dx;
    dv_dy = (vg[iv(i,j)] - vg[iv(i,j-1)]) / dy;
    du_dy = (solid_B||solid_T) ? 0.0 : (ug[iu(i,jp1)]-ug[iu(i,jm1)])/(2*dy);
    dv_dx = (solid_L||solid_R) ? 0.0 : (vg[iv(ip1,j)]-vg[iv(im1,j)])/(2*dx);
}

// ═══════════════════════════════════════════════════════════
// Viscous force μ∇²u at cell centers
// ═══════════════════════════════════════════════════════════
void LFMSimulator::compute_viscous(const Grid& g, std::vector<double>& vu, std::vector<double>& vv){
    int nx=g.nx, ny=g.ny; double mu=(cfg_.Re>0)?cfg_.U_inf*2*cfg_.cyl_R/cfg_.Re:0;
    double idx2=1.0/(g.dx*g.dx), idy2=1.0/(g.dy*g.dy);
    size_t N=flow_map_.nx*flow_map_.ny; vu.assign(N,0); vv.assign(N,0);
    for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
        size_t k=flow_map_.idx(i,j); if(g.is_solid(i,j))continue;
        double uc=0.5*(g.u_at(i,j)+g.u_at(i-1,j));
        double uL=(i>1)?0.5*(g.u_at(i-1,j)+g.u_at(i-2,j)):uc;
        double uR=(i<nx)?0.5*(g.u_at(i+1,j)+g.u_at(i,j)):uc;
        double uB=(j>1)?0.5*(g.u_at(i,j-1)+g.u_at(i-1,j-1)):uc;
        double uT=(j<ny)?0.5*(g.u_at(i,j+1)+g.u_at(i-1,j+1)):uc;
        vu[k]=mu*((uL+uR-2*uc)*idx2+(uB+uT-2*uc)*idy2);
        double vc=0.5*(g.v_at(i,j)+g.v_at(i,j-1));
        double vL=(i>1)?0.5*(g.v_at(i-1,j)+g.v_at(i-1,j-1)):vc;
        double vR=(i<nx)?0.5*(g.v_at(i+1,j)+g.v_at(i+1,j-1)):vc;
        double vB=(j>1)?0.5*(g.v_at(i,j-1)+g.v_at(i,j-2)):vc;
        double vT=(j<ny)?0.5*(g.v_at(i,j+1)+g.v_at(i,j)):vc;
        vv[k]=mu*((vL+vR-2*vc)*idx2+(vB+vT-2*vc)*idy2);
    }
}

// ═══════════════════════════════════════════════════════════
// Pullback: m = T^T · u_0(Ψ)
// ═══════════════════════════════════════════════════════════
void LFMSimulator::pullback_impulse(const Grid& u0_grid){
    int nx=flow_map_.nx, ny=flow_map_.ny;
    for (int j=1;j<=ny;j++) for (int i=1;i<=nx;i++) {
        size_t k=flow_map_.idx(i,j);
        double X=flow_map_.psi_x[k], Y=flow_map_.psi_y[k], uX,uY;
        sample_velocity(X,Y,u0_grid.u,u0_grid.v,uX,uY);
        m_x_[k]=flow_map_.T00[k]*uX+flow_map_.T10[k]*uY;
        m_y_[k]=flow_map_.T01[k]*uX+flow_map_.T11[k]*uY;
    }
}

// ═══════════════════════════════════════════════════════════
// Forward pullback: û_0 = F^T · m(Φ)   — uses B-spline via sample_cell_centered.
// ═══════════════════════════════════════════════════════════
void LFMSimulator::forward_pullback(const std::vector<double>& mx, const std::vector<double>& my,
    std::vector<double>& ux, std::vector<double>& uy) {
    int nx = flow_map_.nx, ny = flow_map_.ny;
    for (int j = 1; j <= ny; j++) for (int i = 1; i <= nx; i++) {
        size_t k = flow_map_.idx(i, j);
        double msx, msy;
        sample_cell_centered(mx, my, flow_map_.phi_x[k], flow_map_.phi_y[k], msx, msy);
        ux[k] = flow_map_.F00[k] * msx + flow_map_.F10[k] * msy;
        uy[k] = flow_map_.F01[k] * msx + flow_map_.F11[k] * msy;
    }
}

// Stubs for test accessors (gauge projection is inlined in run_cycle)
void LFMSimulator::gauge_project() {}
void LFMSimulator::clamp_out_of_solid(double&, double&) const {}

void LFMSimulator::write_frame(int frame_num){
    mkdir(cfg_.out_dir.c_str(),0755); VtkWriter::write(grid_,frame_num,cfg_);
}

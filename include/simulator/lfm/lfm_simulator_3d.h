#pragma once
#include "config/config.h"
#include "core/grid_3d.h"
#include "numerics/bc/patches_3d.h"
#include "simulator/lfm/flow_map_3d.h"
#include "simulator/simulator_3d.h"
#include "solver/solver_3d.h"
#include <memory>
#include <vector>

/// 3D LFM simulator per Algorithm 1 of Sun et al. 2025 (impulse-based).
/// Port of the 2D LFMSimulator (include/simulator/lfm/lfm_simulator.h) onto the
/// Grid3D / Solver3D stack. Like ChorinSimulator3D, the initial condition is
/// injected via mutable_grid() and the BC stack via set_boundary_manager();
/// there is no 3D scenario registry yet.
class LFMSimulator3D : public Simulator3D {
public:
    LFMSimulator3D(const Config& cfg, std::unique_ptr<Solver3D> solver,
                   bc::BoundaryManager3D bcs = bc::free_slip_box());

    // Finalize initialization: apply the wall BCs over the injected initial
    // condition. Mirrors CudaLFMSimulator3D::commit() so the simulator factory
    // drives both backends the same way (construct → setup → commit).
    void commit();

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
    void set_boundary_manager(bc::BoundaryManager3D mgr) {
        bcs_ = std::move(mgr);
        bcs_.apply(grid_);
    }

    void run_cycle(int n_steps);

private:
    Config cfg_;
    Grid3D grid_;
    double t_ = 0;
    int step_ = 0;
    std::unique_ptr<Solver3D> solver_;
    bc::BoundaryManager3D bcs_;
    FlowMap3D flow_map_;

    std::vector<double> m_x_, m_y_, m_z_; // impulse on staggered MAC faces (FIX①)

    // ── FIX①: per-axis (staggered-face) flow maps ──
    // Three per-axis flow maps carry the impulse directly on the MAC faces (zero
    // face↔center averaging). For each face type a∈{u,v,w} we store a backward
    // position ψ_a + Jacobian COVECTOR ROW T_a=(Ta0,Ta1,Ta2) and a forward
    // position φ_a + covector row F_a, all on that axis' face grid. dψ/dt=u(ψ),
    // dT_a/dt=∇u(ψ)·T_a (single 3-vector, ≡ the matching column of the full 3×3
    // Jacobian). Identity covector: u→(1,0,0), v→(0,1,0), w→(0,0,1). Pullback is
    // m_a_face = T_a·u0(ψ_a), landing directly on faces (author's PullbackAxis).
    // Arrays are MAC-sized (u_size/v_size/w_size) so sample_velocity / iu,iv,iw
    // index them with no extra bookkeeping (ghost rows unused).
    struct FaceFlowMap {
        // backward (Ψ, T): pos + covector row
        std::vector<double> bx, by, bz, t0, t1, t2;
        // forward (Φ, F): pos + covector row
        std::vector<double> fx, fy, fz, f0, f1, f2;
    };
    FaceFlowMap fmu_, fmv_, fmw_; // u-, v-, w-face flow maps

    // Midpoint flow-map state for path-integral quadrature
    std::vector<double> phi_mid_x_, phi_mid_y_, phi_mid_z_;
    std::vector<double> F_mid_00_, F_mid_01_, F_mid_02_, F_mid_10_, F_mid_11_, F_mid_12_, F_mid_20_,
        F_mid_21_, F_mid_22_;

    struct VelocitySnapshot {
        std::vector<double> u, v, w;
    };
    std::vector<VelocitySnapshot> vel_buffer_;

    // ---- Algorithm 1 sub-steps ----
    void rk2_advect(Grid3D& dst, const Grid3D& src, const std::vector<double>& vu,
                    const std::vector<double>& vv, const std::vector<double>& vw, double dt_step);
    void project(Grid3D& g);

    void rk4_march_forward(const std::vector<double>& u, const std::vector<double>& v,
                           const std::vector<double>& w, double dt_march);
    void rk4_march_backward(const std::vector<double>& u, const std::vector<double>& v,
                            const std::vector<double>& w, double dt_march);

    void save_flow_map_state();
    void compute_midpoints();

    void compute_viscous(const Grid3D& g, std::vector<double>& vu, std::vector<double>& vv,
                         std::vector<double>& vw);
    void accumulate_to_u0(Grid3D& u0, const std::vector<double>& vu, const std::vector<double>& vv,
                          const std::vector<double>& vw, double coeff);
    void accumulate_path_integral(Grid3D& u0, const std::vector<double>& visc_u,
                                  const std::vector<double>& visc_v,
                                  const std::vector<double>& visc_w, double coeff);

    void pullback_impulse(const Grid3D& u0_grid);
    void forward_pullback(const std::vector<double>& mx, const std::vector<double>& my,
                          const std::vector<double>& mz, std::vector<double>& ux,
                          std::vector<double>& uy, std::vector<double>& uz);

    // ── FIX① per-axis (staggered-face) flow-map operations ──
    // Set each face flow map to identity (position = that face's physical
    // coordinate, covector row = the corresponding unit row e_a).
    void face_set_forward_identity();
    void face_set_backward_identity();
    // One RK4 step of a single face point's (pos, covector) ODE: dp/dt=u(p),
    // dT/dt=∇u(p)·T. axis ∈ {0=u,1=v,2=w} selects the face's MAC offset. Mirrors
    // march_cell's RK4 + ∇u arithmetic but carries only the 3-component covector
    // (one Jacobian column) → bit-identical to extracting that column of march_cell.
    void face_march_point(int axis, double& px, double& py, double& pz, double T[3],
                          const std::vector<double>& u, const std::vector<double>& v,
                          const std::vector<double>& w, double dt_march) const;
    void face_march_forward(const std::vector<double>& u, const std::vector<double>& v,
                            const std::vector<double>& w, double dt_march);
    void face_march_backward(const std::vector<double>& u, const std::vector<double>& v,
                             const std::vector<double>& w, double dt_march);
    // Per-face pullback m_a = T_a·src(ψ_a) on every face (uses backward map by
    // default; fwd=true uses the forward map φ_a/F_a). Writes into the MAC-shaped
    // dst_{u,v,w}, sampling the staggered src field via sample_velocity (no averaging).
    void face_pullback(const std::vector<double>& su, const std::vector<double>& sv,
                       const std::vector<double>& sw, std::vector<double>& dst_u,
                       std::vector<double>& dst_v, std::vector<double>& dst_w, bool fwd) const;
    // Face BFECC error correction + clamp + write m → grid velocity (no gauge avg).
    void face_error_correction(const Grid3D& u0_grid);
    // Clamp m_{u,v,w}_ component to the [min,max] of its 6 same-face-grid
    // neighbours' pre-correction values (author's BfeccClamp, per axis grid).
    void face_bfecc_clamp(const std::vector<double>& pre_u, const std::vector<double>& pre_v,
                          const std::vector<double>& pre_w);

    // BFECC clamp: bound the corrected impulse m_{x,y,z}_ to the [min,max] of the
    // un-corrected (pre) values over each cell's 6 face-neighbours. Paper's
    // BfeccClampKernel — keeps inviscid cycles stable.
    void bfecc_clamp_impulse(const std::vector<double>& pre_x, const std::vector<double>& pre_y,
                             const std::vector<double>& pre_z);

    void sample_cell_centered(const std::vector<double>& sx, const std::vector<double>& sy,
                              const std::vector<double>& sz, double x, double y, double z,
                              double& vx, double& vy, double& vz) const;
    void sample_velocity(double x, double y, double z, const std::vector<double>& u,
                         const std::vector<double>& v, const std::vector<double>& w, double& vu,
                         double& vv, double& vw) const;

    // Velocity AND its 3x3 gradient from the SAME quadratic B-spline (analytic spline
    // derivative). g[3a+b]=∂u_a/∂x_b. Mirrors GPU d_sample_velocity_grad bit-for-bit;
    // replaces the nearest-cell finite-difference velocity_gradient_at in the flow-map
    // march so dF/dt=∇u·F is evolved consistently (FIX②, the author's InterpMacN2Grad).
    void sample_velocity_gradient(double x, double y, double z, const std::vector<double>& u,
                                  const std::vector<double>& v, const std::vector<double>& w,
                                  double& vu, double& vv, double& vw, double g[9]) const;

    void velocity_gradient_at(double x, double y, double z, const std::vector<double>& ug,
                              const std::vector<double>& vg, const std::vector<double>& wg,
                              double g[9]) const;
    void velocity_gradient(int i, int j, int k, const std::vector<double>& ug,
                           const std::vector<double>& vg, const std::vector<double>& wg,
                           double g[9]) const;

    // One RK4 step of the coupled flow-map ODE (dΦ/dt = u, dF/dt = ∇u·F) for a
    // single cell. State is position (px,py,pz) + 3×3 Jacobian F[9] (row-major
    // F[3a+b] = ∂Φ_a/∂X_b), updated in place. Shared by forward (Φ,F) and
    // backward (Ψ,T) marches — the only difference is the sign of dt_march.
    void march_cell(double& px, double& py, double& pz, double F[9],
                    const std::vector<double>& u, const std::vector<double>& v,
                    const std::vector<double>& w, double dt_march) const;

    void apply_bc();

public:
    // ---- Public test accessors (parallel the 2D LFMSimulator) ----
    void compute_viscous_public(const Grid3D& g, std::vector<double>& vu, std::vector<double>& vv,
                                std::vector<double>& vw) {
        compute_viscous(g, vu, vv, vw);
    }
    void sample_velocity_public(double x, double y, double z, const std::vector<double>& u,
                                const std::vector<double>& v, const std::vector<double>& w,
                                double& vu, double& vv, double& vw) const {
        sample_velocity(x, y, z, u, v, w, vu, vv, vw);
    }
    void pullback_impulse_public(const Grid3D& u0) {
        pullback_impulse(u0);
    }
    void rk4_march_forward_public(const std::vector<double>& u, const std::vector<double>& v,
                                  const std::vector<double>& w, double dt) {
        rk4_march_forward(u, v, w, dt);
    }
    void rk4_march_backward_public(const std::vector<double>& u, const std::vector<double>& v,
                                   const std::vector<double>& w, double dt) {
        rk4_march_backward(u, v, w, dt);
    }
    void rk2_advect_public(Grid3D& dst, const Grid3D& src, const std::vector<double>& vu,
                           const std::vector<double>& vv, const std::vector<double>& vw,
                           double dt_step) {
        rk2_advect(dst, src, vu, vv, vw, dt_step);
    }
    void forward_pullback_public(const std::vector<double>& mx, const std::vector<double>& my,
                                 const std::vector<double>& mz, std::vector<double>& ux,
                                 std::vector<double>& uy, std::vector<double>& uz) {
        forward_pullback(mx, my, mz, ux, uy, uz);
    }
    FlowMap3D& flow_map() {
        return flow_map_;
    }
    size_t vel_buffer_size() const {
        return vel_buffer_.size();
    }
    void vel_buffer_public(int i, std::vector<double>& u, std::vector<double>& v,
                           std::vector<double>& w) const {
        u = vel_buffer_[i].u;
        v = vel_buffer_[i].v;
        w = vel_buffer_[i].w;
    }
    const std::vector<double>& impulse_x() const {
        return m_x_;
    }
    const std::vector<double>& impulse_y() const {
        return m_y_;
    }
    const std::vector<double>& impulse_z() const {
        return m_z_;
    }

    // ── FIX① per-face flow-map test accessors ──
    const FaceFlowMap& face_map_u() const { return fmu_; }
    const FaceFlowMap& face_map_v() const { return fmv_; }
    const FaceFlowMap& face_map_w() const { return fmw_; }
    void face_set_forward_identity_public() { face_set_forward_identity(); }
    void face_set_backward_identity_public() { face_set_backward_identity(); }
    void face_march_forward_public(const std::vector<double>& u, const std::vector<double>& v,
                                   const std::vector<double>& w, double dt) {
        face_march_forward(u, v, w, dt);
    }
    void face_march_backward_public(const std::vector<double>& u, const std::vector<double>& v,
                                    const std::vector<double>& w, double dt) {
        face_march_backward(u, v, w, dt);
    }
    void face_pullback_public(const std::vector<double>& su, const std::vector<double>& sv,
                              const std::vector<double>& sw, std::vector<double>& du,
                              std::vector<double>& dv, std::vector<double>& dw, bool fwd) {
        face_pullback(su, sv, sw, du, dv, dw, fwd);
    }
};

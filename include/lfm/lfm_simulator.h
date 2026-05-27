#pragma once
#include "config/config.h"
#include "core/grid.h"
#include "lfm/flow_map_2d.h"
#include "simulator/simulator_base.h"
#include "solver/solver.h"
#include <memory>
#include <vector>

/// LFM simulator per Algorithm 1 of Sun et al. 2025 (impulse-based).
class LFMSimulator : public Simulator {
public:
    LFMSimulator(const Config& cfg, std::unique_ptr<Solver> solver);

    void step() override;
    const Grid& grid() const override { return grid_; }
    double time() const override { return t_; }
    int  step_count() const override { return step_; }
    void run_cycle(int n_steps);

private:
    Config cfg_;
    Grid   grid_;
    double t_ = 0;
    int    step_ = 0;
    std::unique_ptr<Solver> solver_;
    FlowMap2D flow_map_;

    std::vector<double> m_x_, m_y_;

    // Midpoint flow map state for path integral quadrature
    std::vector<double> phi_mid_x_, phi_mid_y_;
    std::vector<double> F_mid_00_, F_mid_10_, F_mid_01_, F_mid_11_;

    struct VelocitySnapshot { std::vector<double> u, v; };
    std::vector<VelocitySnapshot> vel_buffer_;

    // ---- Algorithm 1 sub-steps ----

    void rk2_advect(Grid& dst, const Grid& src,
                    const std::vector<double>& vel_u,
                    const std::vector<double>& vel_v, double dt_step);
    void project(Grid& g);

    void rk4_march_forward(const std::vector<double>& u,
                           const std::vector<double>& v, double dt_march);
    void rk4_march_backward(const std::vector<double>& u,
                            const std::vector<double>& v, double dt_march);

    void save_flow_map_state();
    void compute_midpoints();

    void compute_viscous(const Grid& g, std::vector<double>& vu, std::vector<double>& vv);
    void accumulate_to_u0(Grid& u0, const std::vector<double>& vu,
                          const std::vector<double>& vv, double coeff);
    void accumulate_path_integral(Grid& u0, const std::vector<double>& visc_u,
                                  const std::vector<double>& visc_v, double coeff);

    void pullback_impulse(const Grid& u0_grid);
    void forward_pullback(const std::vector<double>& mx, const std::vector<double>& my,
                          std::vector<double>& ux, std::vector<double>& uy);

    void sample_cell_centered(const std::vector<double>& sx, const std::vector<double>& sy,
                              double x, double y, double& vx, double& vy) const;
    double sample_u(double x, double y, const std::vector<double>& u,
                    const std::vector<double>& v) const;
    double sample_v(double x, double y, const std::vector<double>& u,
                    const std::vector<double>& v) const;
    void sample_velocity(double x, double y,
                         const std::vector<double>& u, const std::vector<double>& v,
                         double& vu, double& vv) const;

    void velocity_gradient_at(double x, double y,
        const std::vector<double>& ug, const std::vector<double>& vg,
        double& du_dx, double& du_dy, double& dv_dx, double& dv_dy) const;
    void velocity_gradient(int i, int j,
                           const std::vector<double>& u, const std::vector<double>& v,
                           double& du_dx, double& du_dy,
                           double& dv_dx, double& dv_dy) const;

    void gauge_project();
    void clamp_out_of_solid(double& x, double& y) const;
    void write_frame(int frame_num);

public:
    // ---- Public test accessors ----
    void compute_viscous_public(const Grid& g, std::vector<double>& vu, std::vector<double>& vv)
        { compute_viscous(g, vu, vv); }
    void sample_velocity_public(double x, double y, const std::vector<double>& u,
                                const std::vector<double>& v, double& vu, double& vv) const
        { sample_velocity(x, y, u, v, vu, vv); }
    void gauge_project_public() { gauge_project(); }
    void pullback_impulse_public(const Grid& u0) { pullback_impulse(u0); }
    void rk4_march_forward_public(const std::vector<double>& u, const std::vector<double>& v, double dt)
        { rk4_march_forward(u, v, dt); }
    void rk4_march_backward_public(const std::vector<double>& u, const std::vector<double>& v, double dt)
        { rk4_march_backward(u, v, dt); }
    void set_impulse_uniform_public(double mx, double my) {
        std::fill(m_x_.begin(), m_x_.end(), mx);
        std::fill(m_y_.begin(), m_y_.end(), my);
    }
    FlowMap2D& flow_map() { return flow_map_; }
    const std::vector<double>& impulse_x() const { return m_x_; }
    const std::vector<double>& impulse_y() const { return m_y_; }
};

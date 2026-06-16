#pragma once
#include <array>
#include <map>
#include <string>

// ── Unified 2D / 3D configuration ──
// `dim` selects the dimensionality; 3D fields (NZ, Lz, cyl_cz) are ignored
// when dim == 2. The simulator factory dispatches on `dim`.
class Config {
public:
    // ── Dimensionality ──
    int dim = 2;                         // 2 or 3
    int NX = 128, NY = 32, NZ = 1;       // NZ used only when dim == 3
    double Lx = 4.0, Ly = 1.0, Lz = 1.0; // Lz used only when dim == 3

    // ── Scenario ──
    std::string scenario = "karman";
    double U_inf         = 1.0;
    double Re            = 200.0;
    double cyl_cx = 1.0, cyl_cy = 0.5, cyl_cz = 0.5;
    double cyl_R              = 0.1;
    std::string cylinder_type = "stair"; // "stair" | "smooth"

    // ── Time integration ──
    std::string time_integrator = "chorin"; // "chorin" | "lfm"
    double dt                   = 0.005;
    double t_end                = 10.0;
    int lfm_cycle_steps         = 10;    // n in LFM Algorithm 1
    bool lfm_bfecc_clamp        = false; // clamp BFECC-corrected impulse to neighbor min/max
    bool lfm_march_fp32         = true;  // FP32 face flow-map marching/sampling (author-faithful, ~order faster on consumer GPUs)
                                         // (paper's BfeccClamp: lets inviscid runs stay stable)
    // Velocity wall BC for the LFM cycle: "free_slip" (closed box, default) or
    // "freestream" (prescribe inflow_u on all walls — delta wing / wind tunnel).
    std::string lfm_bc = "free_slip";
    double inflow_ux = 0.0, inflow_uy = 0.0, inflow_uz = 0.0;

    // ── Pressure solver ──
    std::string solver = "pcg"; // jacobi | rbgs | cg | pcg | pcg_gmg | pcg_amg | pcg_uaamg
    int solve_iters    = 2000;
    double solve_tol   = 1e-6;

    // ── I/O ──
    int frame_skip      = 10;
    std::string out_dir = "output";

    // ── Scenario-specific extras ──
    // Keys from an INPUT file that don't map to a core field above land here
    // (e.g. 3D geometry knobs: ring_radius, blade_omega, buoyancy_beta, …).
    // Read them through the typed accessors below; each falls back to `def`
    // when the key is absent or unparseable.
    std::map<std::string, std::string> extra;

    double dget(const std::string& key, double def) const;
    int iget(const std::string& key, int def) const;
    std::string sget(const std::string& key, const std::string& def) const;
    // Parse a comma-separated triple ("0.5,0.5,0.5") into {x,y,z}.
    std::array<double, 3> v3get(const std::string& key, std::array<double, 3> def) const;
};

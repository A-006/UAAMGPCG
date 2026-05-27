#pragma once
#include <string>

// ── Unified 2D / 3D configuration ──
// `dim` selects the dimensionality; 3D fields (NZ, Lz, cyl_cz) are ignored
// when dim == 2. The simulator factory dispatches on `dim`.
class Config {
public:
    // ── Dimensionality ──
    int    dim = 2;                          // 2 or 3
    int    NX = 128, NY = 32, NZ = 1;        // NZ used only when dim == 3
    double Lx = 4.0, Ly = 1.0, Lz = 1.0;     // Lz used only when dim == 3

    // ── Scenario ──
    std::string scenario = "karman";
    double U_inf = 1.0;
    double Re = 200.0;
    double cyl_cx = 1.0, cyl_cy = 0.5, cyl_cz = 0.5;
    double cyl_R = 0.1;
    std::string cylinder_type = "stair";     // "stair" | "smooth"

    // ── Time integration ──
    std::string time_integrator = "chorin";  // "chorin" | "lfm"
    double dt = 0.005;
    double t_end = 10.0;
    int    lfm_cycle_steps = 10;             // n in LFM Algorithm 1

    // ── Pressure solver ──
    std::string solver = "pcg";              // jacobi | rbgs | cg | pcg | pcg_gmg | pcg_amg | pcg_uaamg
    int    solve_iters = 2000;
    double solve_tol   = 1e-6;

    // ── I/O ──
    int    frame_skip = 10;
    std::string out_dir = "output";
};

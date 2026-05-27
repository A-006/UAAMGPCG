#pragma once
#include <string>

class Config {
public:
    std::string scenario = "karman";
    int    NX = 128, NY = 32;
    double Lx = 4.0, Ly = 1.0;
    double U_inf = 1.0;
    double cyl_cx = 1.0, cyl_cy = 0.5, cyl_R = 0.1;
    double Re = 200.0;
    double dt = 0.005;
    double t_end = 10.0;
    int    frame_skip = 10;
    int    solve_iters = 2000;
    double solve_tol  = 1e-6;
    std::string out_dir = "output";
    std::string solver = "pcg";
    std::string time_integrator = "chorin";  // "chorin" or "lfm"
    int    lfm_cycle_steps = 10;            // steps per LFM reinitialization cycle
    std::string cylinder_type = "stair";    // "stair" or "smooth"
};

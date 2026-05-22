#pragma once
#include <string>

enum class Solver { Jacobi, RBGS, CG, PCG };

inline const char* solver_name(Solver s) {
    switch (s) {
        case Solver::Jacobi: return "Jacobi";
        case Solver::RBGS:   return "RBGS";
        case Solver::CG:     return "CG";
        case Solver::PCG:    return "PCG";
    }
    return "?";
}

struct Config {
    std::string scenario = "karman";
    int    NX = 128, NY = 32;
    double Lx = 4.0, Ly = 1.0;
    double U_inf = 1.0;
    double cyl_cx = 1.0, cyl_cy = 0.5, cyl_R = 0.1;
    double Re = 200.0;
    double dt = 0.005;
    double t_end = 10.0;
    int    frame_skip = 10;
    int    solve_iters = 2000;        // iterations for Jacobi/RBGS; max for CG/PCG
    double solve_tol  = 1e-6;        // tolerance for CG/PCG
    Solver solver = Solver::Jacobi;
    std::string out_dir = "output";
};

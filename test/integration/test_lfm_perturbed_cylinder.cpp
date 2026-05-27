/**
 * @file test_lfm_perturbed_cylinder.cpp
 * @brief Simplest Karman test: start with an asymmetric vortex behind cylinder.
 *
 * If LFM can sustain vortex shedding from a large initial perturbation,
 * the algorithm works but needs a trigger. If not, the algorithm itself
 * damps all unsteady structures.
 *
 * Build: add to test/CMakeLists.txt
 * Run:   ./build/test/test_lfm_perturbed_cylinder
 */
#include "config/config.h"
#include "core/grid.h"
#include "lfm/lfm_simulator.h"
#include "solver/factory.h"
#include "boundary/boundary.h"
#include "force/force.h"
#include "../test_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>

// Add a strong clockwise vortex behind the cylinder (top side)
static void add_wake_vortex(Grid& g, double cx, double cy, double strength, double radius) {
    for (int i=0;i<=g.nx;i++) for (int j=1;j<=g.ny;j++) {
        if(g.is_solid(i,j)||g.is_solid(i+1,j))continue;
        double x=i*g.dx, y=(j-0.5)*g.dy;
        double rx=x-cx, ry=y-cy, r=std::sqrt(rx*rx+ry*ry);
        if(r<1e-12)continue;
        double u_th = strength/(2*M_PI*r)*(1-std::exp(-r*r/(radius*radius)));
        g.u_at(i,j) -= u_th * ry/r;
    }
    for (int i=1;i<=g.nx;i++) for (int j=0;j<=g.ny;j++) {
        if(g.is_solid(i,j)||g.is_solid(i,j+1))continue;
        double x=(i-0.5)*g.dx, y=j*g.dy;
        double rx=x-cx, ry=y-cy, r=std::sqrt(rx*rx+ry*ry);
        if(r<1e-12)continue;
        double u_th = strength/(2*M_PI*r)*(1-std::exp(-r*r/(radius*radius)));
        g.v_at(i,j) += u_th * rx/r;
    }
}

int main() {
    test_header("Perturbed Cylinder: strong wake vortex behind cylinder");

    Config cfg; cfg.NX=128; cfg.NY=32; cfg.Lx=4.0; cfg.Ly=1.0;
    cfg.Re=200; cfg.cyl_cx=1.0; cfg.cyl_cy=0.5; cfg.cyl_R=0.1;
    cfg.scenario="karman"; cfg.U_inf=1.0;
    cfg.dt=0.25*(cfg.Lx/cfg.NX)/cfg.U_inf;
    cfg.solve_iters=200; cfg.solve_tol=1e-8;
    cfg.time_integrator="lfm"; cfg.lfm_cycle_steps=2;
    cfg.out_dir="/tmp/karman_perturbed";
    mkdir(cfg.out_dir.c_str(), 0755);
    auto solver = Factory::create("pcg_uaamg");
    LFMSimulator sim(cfg, std::move(solver));
    Grid& g = const_cast<Grid&>(sim.grid());

    // Add asymmetric vortex behind cylinder (top side only — breaks symmetry)
    add_wake_vortex(g, 1.3, 0.65, -3.0, 0.08);  // CW, stronger
    add_wake_vortex(g, 1.3, 0.35, +1.5, 0.08);  // CCW, weaker — asymmetry!

    double D=2*cfg.cyl_R, U=cfg.U_inf;
    std::cout<<"  cycle  Cl      Cd      max|w|   max|div|\n";

    int ncycles=200;
    for(int c=0;c<ncycles;c++){
        sim.step();
        auto force=computeForce(sim.grid(),cfg.dt,U,cfg.Re,cfg.cyl_cx,cfg.cyl_cy,cfg.cyl_R);
        double mw=0, md=0;
        for(int i=1;i<=cfg.NX;i++) for(int j=1;j<=cfg.NY;j++){
            if(sim.grid().is_solid(i,j))continue;
            double w=std::abs((sim.grid().v_at(i,j)-sim.grid().v_at(i-1,j))/cfg.dt
                             -(sim.grid().u_at(i,j)-sim.grid().u_at(i,j-1))/cfg.dt);
            mw=std::max(mw,w);
            md=std::max(md,std::abs(sim.grid().divergence(i,j)));
        }
        if(c%20==0||c==ncycles-1)
            std::cout<<"  "<<std::setw(4)<<c<<"  "<<std::fixed<<std::setprecision(3)
                     <<force.Cl(U,D)<<"  "<<force.Cd(U,D)<<"  "<<std::setprecision(1)
                     <<mw<<"  "<<std::scientific<<md<<"\n";
    }

    std::cout<<"\n  If Cl oscillates after cycle ~40: LFM CAN sustain shedding (needs trigger)\n";
    std::cout<<"  If Cl stays constant: LFM damps ALL unsteady structures\n";
    return 0;
}

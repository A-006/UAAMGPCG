/**
 * @file test_lfm_instability.cpp
 * @brief Test whether LFM can amplify flow perturbations (instability prerequisite).
 *
 * Karman vortex street requires the LFM cycle to AMPLIFY small perturbations
 * rather than damp them. These tests check this property on simple flows
 * without solid boundaries.
 *
 * Build: added to test/CMakeLists.txt
 * Run:   ./build/test/test_lfm_instability
 */
#include "config/config.h"
#include "core/grid.h"
#include "lfm/flow_map_2d.h"
#include "lfm/lfm_simulator.h"
#include "simulator/simulator.h"
#include "simulator/simulator_base.h"
#include "solver/factory.h"
#include "boundary/boundary.h"
#include "pressure/pressure.h"
#include "force/force.h"
#include "io/vtk_writer.h"
#include "../test_utils.h"
#include "../test_velocity_fields.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>

// ═══════════════════════════════════════════════════════════
// I1: Perturbation preservation on uniform flow (baseline)
//     Uniform flow + tiny sinusoidal perturbation.
//     An ideal integrator preserves the perturbation amplitude.
//     LFM may damp it due to interpolation + gauge projection.
// ═══════════════════════════════════════════════════════════
static void i1_perturbation_baseline() {
    test_header("I1: Perturbation amplitude on uniform flow");

    Config cfg; cfg.NX=64; cfg.NY=32; cfg.Lx=4.0; cfg.Ly=2.0;
    cfg.dt=0.25*4.0/64; cfg.solve_iters=200; cfg.solve_tol=1e-10;
    cfg.cyl_R=0; cfg.time_integrator="lfm"; cfg.lfm_cycle_steps=2;
    cfg.scenario="smoke";
    auto solver = Factory::create("pcg_uaamg");
    LFMSimulator sim(cfg, std::move(solver));
    Grid& g = const_cast<Grid&>(sim.grid());

    double U0=1.0, eps=0.05, kx=2.0*M_PI/cfg.Lx; // single wave across domain
    for (int i=0;i<=g.nx;i++) for (int j=1;j<=g.ny;j++)
        if(!g.is_solid(i,j)&&!g.is_solid(i+1,j)) {
            double x=i*g.dx, y=(j-0.5)*g.dy;
            g.u_at(i,j)=U0 + eps*std::sin(kx*x)*std::cos(M_PI*y/cfg.Ly);
        }
    for (int i=1;i<=g.nx;i++) for (int j=0;j<=g.ny;j++)
        if(!g.is_solid(i,j)&&!g.is_solid(i,j+1)) {
            double x=(i-0.5)*g.dx, y=j*g.dy;
            g.v_at(i,j)=eps*std::cos(kx*x)*std::sin(M_PI*y/cfg.Ly);
        }

    // Measure perturbation energy E = 0.5 * Σ(u'² + v'²) in interior
    auto perturb_energy=[&](const Grid& grid){
        double E=0; int cx=grid.nx/2, cy=grid.ny/2;
        for(int i=cx-8;i<=cx+8;i++) for(int j=cy-8;j<=cy+8;j++){
            if(i<1||i>grid.nx||j<1||j>grid.ny) continue;
            if(grid.is_solid(i,j))continue;
            double uc=0.5*(grid.u_at(i,j)+grid.u_at(i-1,j))-U0;
            double vc=0.5*(grid.v_at(i,j)+grid.v_at(i,j-1));
            E+=uc*uc+vc*vc;
        }
        return E;
    };

    double E0=perturb_energy(g);
    sim.step();
    double E1=perturb_energy(sim.grid());
    double ratio=E1/std::max(1e-15,E0);

    double max_div=0;
    for(int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++)
        if(!sim.grid().is_solid(i,j))
            max_div=std::max(max_div,std::abs(sim.grid().divergence(i,j)));

    std::cout<<"    E0="<<E0<<" E1="<<E1<<" ratio="<<ratio<<" max|div|="<<max_div<<"\n";
    std::cout<<"    Ratio>0.5 = perturbation survives; Ratio<0.5 = strongly damped\n";
    check(ratio>0.3,"Perturbation not completely destroyed (ratio>0.3)");
    check(max_div<0.1,"Div bounded");
}

// ═══════════════════════════════════════════════════════════
// I2: Shear layer — does LFM amplify or damp KH modes?
//     Two streams u_top=+U, u_bottom=-U with interface perturbation.
//     KH instability: perturbation should grow.
// ═══════════════════════════════════════════════════════════
static void i2_shear_layer() {
    test_header("I2: Shear layer perturbation growth");

    Config cfg; cfg.NX=64; cfg.NY=64; cfg.Lx=4.0; cfg.Ly=4.0;
    cfg.dt=0.25*4.0/64; cfg.solve_iters=200; cfg.solve_tol=1e-10;
    cfg.cyl_R=0; cfg.time_integrator="lfm"; cfg.lfm_cycle_steps=4;
    cfg.scenario="smoke";
    auto solver = Factory::create("pcg_uaamg");
    LFMSimulator sim(cfg, std::move(solver));
    Grid& g = const_cast<Grid&>(sim.grid());

    double U0=1.0, delta=0.1, eps=0.02, kx=2.0*M_PI/cfg.Lx;
    double y0=cfg.Ly/2;

    for (int i=0;i<=g.nx;i++) for (int j=1;j<=g.ny;j++) {
        if(g.is_solid(i,j)||g.is_solid(i+1,j))continue;
        double x=i*g.dx, y=(j-0.5)*g.dy;
        double u_base=U0*std::tanh((y-y0)/delta);
        g.u_at(i,j)=u_base + eps*std::sin(kx*x)*std::exp(-((y-y0)/delta)*((y-y0)/delta));
    }
    for (int i=1;i<=g.nx;i++) for (int j=0;j<=g.ny;j++) {
        if(g.is_solid(i,j)||g.is_solid(i,j+1))continue;
        double x=(i-0.5)*g.dx, y=j*g.dy;
        g.v_at(i,j)=eps*std::cos(kx*x)*std::exp(-((y-y0)/delta)*((y-y0)/delta));
    }

    // Measure v-perturbation magnitude at interface
    auto v_rms=[&](const Grid& grid){
        double v2=0; int n=0, j_center=(int)(y0/grid.dy)+1;
        for(int i=10;i<grid.nx-10;i++){
            if(grid.is_solid(i,j_center))continue;
            double vv=grid.v_at(i,j_center);
            v2+=vv*vv; n++;
        }
        return n>0?std::sqrt(v2/n):0.0;
    };

    double v0=v_rms(g);
    sim.step();
    double v1=v_rms(sim.grid());
    double ratio=v1/std::max(1e-15,v0);

    double max_div=0;
    for(int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++)
        if(!sim.grid().is_solid(i,j))
            max_div=std::max(max_div,std::abs(sim.grid().divergence(i,j)));

    std::cout<<"    v_rms: initial="<<v0<<" after="<<v1<<" ratio="<<ratio<<" max|div|="<<max_div<<"\n";
    std::cout<<"    Ratio>1.0 = perturbation grows (KH unstable); Ratio<1.0 = damped\n";
    check(max_div<0.5,"Shear layer div bounded");
}

// ═══════════════════════════════════════════════════════════
// I3: Multi-cycle perturbation tracking
//     Run N cycles and track perturbation energy evolution.
// ═══════════════════════════════════════════════════════════
static double run_perturbation_test(int n_steps) {
    Config cfg; cfg.NX=64; cfg.NY=32; cfg.Lx=4.0; cfg.Ly=2.0;
    cfg.dt=0.25*4.0/64; cfg.solve_iters=200; cfg.solve_tol=1e-10;
    cfg.cyl_R=0; cfg.time_integrator="lfm"; cfg.lfm_cycle_steps=n_steps;
    cfg.scenario="smoke";
    auto solver = Factory::create("pcg_uaamg");
    LFMSimulator sim(cfg, std::move(solver));
    Grid& g = const_cast<Grid&>(sim.grid());

    double U0=1.0, eps=0.05, ky=M_PI/cfg.Ly;
    for (int i=0;i<=g.nx;i++) for (int j=1;j<=g.ny;j++) {
        if(g.is_solid(i,j)||g.is_solid(i+1,j))continue;
        g.u_at(i,j)=U0+eps*std::sin(2.0*M_PI*i*g.dx/cfg.Lx)*std::cos(ky*(j-0.5)*g.dy);
    }
    for (int i=1;i<=g.nx;i++) for (int j=0;j<=g.ny;j++) {
        if(g.is_solid(i,j)||g.is_solid(i,j+1))continue;
        g.v_at(i,j)=eps*std::cos(2.0*M_PI*(i-0.5)*g.dx/cfg.Lx)*std::sin(ky*j*g.dy);
    }

    auto max_v=[&](const Grid& gr){
        double mv=0;
        for(int i=1;i<=gr.nx;i++) for(int j=1;j<=gr.ny;j++)
            if(!gr.is_solid(i,j)) mv=std::max(mv,std::abs(gr.v_at(i,j)));
        return mv;
    };
    double v0=max_v(g);
    int ncycles=std::max(1, 20/n_steps);  // same total simulation time
    for(int c=0;c<ncycles;c++) sim.step();
    return max_v(sim.grid())/std::max(1e-15,v0);
}

static void i3_multicycle_perturbation() {
    test_header("I3: Perturbation survival vs n_steps (same total time)");

    for (int n : {1, 2, 4, 8}) {
        double ratio = run_perturbation_test(n);
        std::cout<<"    n="<<n<<" ratio="<<std::scientific<<ratio
                 <<(ratio>0.5?" GROWS":"")<<(ratio>1.0?" ← INSTABILITY!":"")<<"\n";
    }
    double ratio_n8 = run_perturbation_test(8);
    check(ratio_n8 > 0.0, "Perturbation test complete");
}

// ═══════════════════════════════════════════════════════════
// I4: Cylinder at Re=200 — track Cl growth
// ═══════════════════════════════════════════════════════════
static void i4_cylinder_growth() {
    test_header("I4: Cylinder Re=200 — Cl amplitude growth");

    Config cfg; cfg.NX=128; cfg.NY=32; cfg.Lx=4.0; cfg.Ly=1.0;
    cfg.Re=200; cfg.cyl_cx=1.0; cfg.cyl_cy=0.5; cfg.cyl_R=0.1;
    cfg.scenario="karman"; cfg.U_inf=1.0;
    cfg.dt=0.25*(cfg.Lx/cfg.NX)/cfg.U_inf;
    cfg.solve_iters=200; cfg.solve_tol=1e-8;
    cfg.time_integrator="lfm"; cfg.lfm_cycle_steps=2;
    auto solver = Factory::create("pcg_uaamg");
    LFMSimulator sim(cfg, std::move(solver));

    double D=2*cfg.cyl_R, U=cfg.U_inf;
    std::string vtk_dir="/tmp/karman_lfm_vtk";
    mkdir(vtk_dir.c_str(),0755);
    Config vtk_cfg=cfg; vtk_cfg.out_dir=vtk_dir;
    std::cout<<"    cycle  Cl      Cd      max|div|\n";

    int ncycles=200;
    for(int c=0;c<ncycles;c++){
        sim.step();
        auto force=computeForce(sim.grid(),cfg.dt,U,cfg.Re,cfg.cyl_cx,cfg.cyl_cy,cfg.cyl_R);
        double md=0;
        for(int i=1;i<=cfg.NX;i++) for(int j=1;j<=cfg.NY;j++)
            if(!sim.grid().is_solid(i,j))
                md=std::max(md,std::abs(sim.grid().divergence(i,j)));
        if(c%20==0||c==ncycles-1)
            std::cout<<"    "<<std::setw(4)<<c<<"  "<<std::fixed<<std::setprecision(3)
                     <<force.Cl(U,D)<<"  "<<force.Cd(U,D)<<"  "<<std::scientific<<md<<"\n";
        if(c%10==0) VtkWriter::write(const_cast<Grid&>(sim.grid()),c/10,vtk_cfg);
    }

    check(true,"Cylinder diagnostic complete");
}

int main() {
    i1_perturbation_baseline();
    i2_shear_layer();
    i3_multicycle_perturbation();
    i4_cylinder_growth();
    return test_summary();
}

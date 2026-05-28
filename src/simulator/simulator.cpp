#include "simulator/simulator.h"
#include "advection/advection.h"
#include "boundary/boundary.h"
#include "pressure/pressure.h"
#include "scenarios/karman.h"
#include "io/vtk_writer.h"
#include "solver/factory.h"
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <chrono>
#include <iomanip>

ChorinSimulator::ChorinSimulator(const Config& cfg, std::unique_ptr<Solver> solver)
    : cfg_(cfg), grid_(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly),
      prev_(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly), solver_(std::move(solver))
{
    if (cfg_.scenario == "karman") {
        scenarios::Karman k{ cfg_.cyl_cx, cfg_.cyl_cy, cfg_.cyl_R, cfg_.U_inf };
        if (k.cyl_R > 0) scenarios::setup_karman_cylinder(grid_, k);
        scenarios::set_uniform_inflow(grid_, k.U_inf);
        // Break the y-symmetry — without this seed a perfectly symmetric setup
        // produces a standing symmetric vortex pair, never the alternating street.
        scenarios::seed_wake_perturbation(grid_, k);
    }
    apply_bc();
    prev_ = grid_;
}

void ChorinSimulator::apply_bc() {
    if (cfg_.scenario == "karman") BoundaryConditions::applyKarman(grid_, cfg_.U_inf);
    else                           BoundaryConditions::applySmoke(grid_);
    BoundaryConditions::applySolid(grid_);
}

void ChorinSimulator::apply_forces() {
    if (cfg_.scenario != "smoke") return;
    for (int i=1;i<=grid_.nx;i++) for(int j=1;j<=grid_.ny;j++)
        if(!grid_.is_solid(i,j)) grid_.v_at(i,j)+=cfg_.dt*5.0;
}

void ChorinSimulator::advect() {
    Grid g_adv = prev_;
    for (int i=1;i<grid_.nx;i++) for(int j=1;j<=grid_.ny;j++) g_adv.u_at(i,j)=0;
    for (int i=1;i<=grid_.nx;i++) for(int j=1;j<grid_.ny;j++) g_adv.v_at(i,j)=0;
    AdvectionScheme::advect(prev_, g_adv, cfg_.dt);
    for (int i=1;i<grid_.nx;i++) for(int j=1;j<=grid_.ny;j++) grid_.u_at(i,j)=g_adv.u_at(i,j);
    for (int i=1;i<=grid_.nx;i++) for(int j=1;j<grid_.ny;j++) grid_.v_at(i,j)=g_adv.v_at(i,j);
}

void ChorinSimulator::project() {
    PressureProjection::project(grid_, cfg_.dt, *solver_, cfg_.solve_iters, cfg_.solve_tol);
}

void ChorinSimulator::diffuse() {
    if (cfg_.Re <= 0) return;
    double nu = cfg_.U_inf * 2.0 * cfg_.cyl_R / cfg_.Re;
    if (nu <= 0) return;
    int nx=grid_.nx, ny=grid_.ny;
    double idx2=1.0/(grid_.dx*grid_.dx), idy2=1.0/(grid_.dy*grid_.dy);
    for (int i=1;i<nx;i++) for(int j=1;j<=ny;j++) {
        if(grid_.is_solid(i,j)||grid_.is_solid(i+1,j))continue;
        double uC=grid_.u_at(i,j), uL=(i>1)?grid_.u_at(i-1,j):0.0, uR=(i<nx-1)?grid_.u_at(i+1,j):0.0;
        if(i>=nx-1)uR=uC;
        bool bf=(j>1&&!grid_.is_solid(i,j-1)&&!grid_.is_solid(i+1,j-1));
        bool tf=(j<ny&&!grid_.is_solid(i,j+1)&&!grid_.is_solid(i+1,j+1));
        double uB=bf?grid_.u_at(i,j-1):0.0, uT=tf?grid_.u_at(i,j+1):0.0;
        if(j==1&&!bf)uB=uC; if(j==ny&&!tf)uT=uC;
        grid_.u_at(i,j)+=cfg_.dt*nu*((uL+uR-2*uC)*idx2+(uB+uT-2*uC)*idy2);
    }
    for (int i=1;i<=nx;i++) for(int j=1;j<ny;j++) {
        if(grid_.is_solid(i,j)||grid_.is_solid(i,j+1))continue;
        double vC=grid_.v_at(i,j), vB=(j>1)?grid_.v_at(i,j-1):0.0, vT=(j<ny-1)?grid_.v_at(i,j+1):0.0;
        if(j>=ny-1)vT=vC;
        bool lf=(i>1&&!grid_.is_solid(i-1,j)&&!grid_.is_solid(i-1,j+1));
        bool rf=(i<nx&&!grid_.is_solid(i+1,j)&&!grid_.is_solid(i+1,j+1));
        double vL=lf?grid_.v_at(i-1,j):0.0, vR=rf?grid_.v_at(i+1,j):0.0;
        if(i==1&&!lf)vL=vC; if(i==nx&&!rf)vR=vC;
        grid_.v_at(i,j)+=cfg_.dt*nu*((vL+vR-2*vC)*idx2+(vB+vT-2*vC)*idy2);
    }
}

void ChorinSimulator::step() {
    apply_forces(); advect(); diffuse(); apply_bc(); project(); apply_bc();
    prev_ = grid_; t_ += cfg_.dt; step_++;
}

void ChorinSimulator::run() {
    mkdir(cfg_.out_dir.c_str(),0755);
    std::cout<<"+"<<std::string(52,'-')<<"+\n";
    std::cout<<"| LFM 2D Fluid — "<<cfg_.scenario<<"  grid:"<<cfg_.NX<<"x"<<cfg_.NY
             <<"  dt="<<cfg_.dt<<"  t_end="<<cfg_.t_end<<" |\n";
    std::cout<<"+"<<std::string(52,'-')<<"+\n\n";
    int nsteps=(int)(cfg_.t_end/cfg_.dt);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int s=0;s<nsteps;s++){step();if(s%cfg_.frame_skip==0){VtkWriter::printStatus(s,t_,grid_);VtkWriter::write(grid_,s/cfg_.frame_skip,cfg_);}}
    auto t1=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double>(t1-t0).count();
    std::cout<<"\nDone: "<<nsteps<<" steps in "<<elapsed<<" s ("<<(elapsed/nsteps*1000)<<" ms/step)\n";
}

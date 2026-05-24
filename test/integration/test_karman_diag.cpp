#include "config/config.h"
#include "core/grid.h"
#include "simulator/simulator.h"
#include "solver/factory.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    Config cfg;
    cfg.scenario = "karman"; cfg.NX=128; cfg.Lx=4.0; cfg.Ly=1.0;
    cfg.U_inf=1.0; cfg.Re=200; cfg.cyl_cx=1.0; cfg.cyl_cy=0.5; cfg.cyl_R=0.1;
    cfg.t_end=0.1; cfg.dt=0.0; cfg.solve_iters=50; cfg.solve_tol=1e-6;
    cfg.frame_skip=100; cfg.out_dir="/tmp/kdiag";
    cfg.NY=std::max(16,cfg.NX/4);
    cfg.dt=0.5*(cfg.Lx/cfg.NX)/cfg.U_inf;
    cfg.solver="pcg_uaamg";

    std::cout << "dt=" << cfg.dt << " grid=" << cfg.NX << "x" << cfg.NY << "\n";

    auto solver = Factory::create(cfg.solver);
    LFMSimulator sim(cfg, std::move(solver));
    int nsteps = (int)(cfg.t_end/cfg.dt);

    for(int s=0;s<nsteps;s++){
        sim.step();
        const Grid& g=sim.grid();
        double max_div=0, max_p=0, max_u=0;
        int nfluid=0;
        for(int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++){
            if(!g.is_solid(i,j)){
                max_div=std::max(max_div,std::abs(g.divergence(i,j)));
                max_p=std::max(max_p,std::abs(g.p_at(i,j)));
                max_u=std::max(max_u,std::abs(g.u_at(i,j)));
                nfluid++;
            }
        }
        double t=(s+1)*cfg.dt;
        std::cout << "step=" << s << " t=" << t
                  << " max_div=" << std::scientific << std::setprecision(2) << max_div
                  << " max_p=" << max_p << " max_u=" << max_u
                  << " nfluid=" << nfluid << "\n";
    }
    return 0;
}

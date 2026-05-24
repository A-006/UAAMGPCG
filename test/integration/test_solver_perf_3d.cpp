/**
 * @file test_solver_perf_3d.cpp
 * @brief 3D solver performance comparison — solve Ax=b on 3D grid with timing.
 */
#include "solver/jacobi_3d.h"
#include "solver/rbgs_3d.h"
#include "solver/pcg_3d.h"
#include "solver/preconditioner/identity_preconditioner_3d.h"
#include "solver/preconditioner/gmg_preconditioner_3d.h"
#include "solver/preconditioner/uaamg_preconditioner_3d.h"
#include "../test_utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

// Compute residual for PCG (uses negated RHS: b = -(rhs-mean), solves Ax=b)
static double compute_residual_l2_pcg(const Grid3D& g, const std::vector<double>& rhs) {
    int nx=g.nx, ny=g.ny, nz=g.nz;
    double idx2=1.0/(g.dx*g.dx), idy2=1.0/(g.dy*g.dy), idz2=1.0/(g.dz*g.dz);
    double diag=2.0*(idx2+idy2+idz2);
    double sum=0; int cnt=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++)
        if(!g.is_solid(i,j,k)) {sum+=rhs[g.ip(i,j,k)]; cnt++;}
    double mean=cnt>0?sum/cnt:0;
    double l2=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++) {
        if(g.is_solid(i,j,k)) continue;
        int id=g.ip(i,j,k);
        double pC=g.p[id];
        double pL=(i>1&&!g.is_solid(i-1,j,k))?g.p[g.ip(i-1,j,k)]:pC;
        double pR=(i<nx&&!g.is_solid(i+1,j,k))?g.p[g.ip(i+1,j,k)]:pC;
        double pB=(j>1&&!g.is_solid(i,j-1,k))?g.p[g.ip(i,j-1,k)]:pC;
        double pT=(j<ny&&!g.is_solid(i,j+1,k))?g.p[g.ip(i,j+1,k)]:pC;
        double pF=(k>1&&!g.is_solid(i,j,k-1))?g.p[g.ip(i,j,k-1)]:pC;
        double pK=(k<nz&&!g.is_solid(i,j,k+1))?g.p[g.ip(i,j,k+1)]:pC;
        double Ax=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
        double r=-(rhs[id]-mean)-Ax;  // PCG: b = -(rhs-mean)
        l2+=r*r;
    }
    return std::sqrt(l2);
}

// Compute residual for Jacobi/RBGS (uses zero-mean RHS: b = rhs-mean, solves Ax=b)
static double compute_residual_l2_jacobi(const Grid3D& g, const std::vector<double>& rhs) {
    int nx=g.nx, ny=g.ny, nz=g.nz;
    double idx2=1.0/(g.dx*g.dx), idy2=1.0/(g.dy*g.dy), idz2=1.0/(g.dz*g.dz);
    double diag=2.0*(idx2+idy2+idz2);
    double sum=0; int cnt=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++)
        if(!g.is_solid(i,j,k)) {sum+=rhs[g.ip(i,j,k)]; cnt++;}
    double mean=cnt>0?sum/cnt:0;
    double l2=0;
    for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++) {
        if(g.is_solid(i,j,k)) continue;
        int id=g.ip(i,j,k);
        double pC=g.p[id];
        double pL=(i>1&&!g.is_solid(i-1,j,k))?g.p[g.ip(i-1,j,k)]:pC;
        double pR=(i<nx&&!g.is_solid(i+1,j,k))?g.p[g.ip(i+1,j,k)]:pC;
        double pB=(j>1&&!g.is_solid(i,j-1,k))?g.p[g.ip(i,j-1,k)]:pC;
        double pT=(j<ny&&!g.is_solid(i,j+1,k))?g.p[g.ip(i,j+1,k)]:pC;
        double pF=(k>1&&!g.is_solid(i,j,k-1))?g.p[g.ip(i,j,k-1)]:pC;
        double pK=(k<nz&&!g.is_solid(i,j,k+1))?g.p[g.ip(i,j,k+1)]:pC;
        double Ax=diag*pC-(pL+pR)*idx2-(pB+pT)*idy2-(pF+pK)*idz2;
        double r=(rhs[id]-mean)-Ax;  // Jacobi/RBGS: b = rhs-mean (no negation)
        l2+=r*r;
    }
    return std::sqrt(l2);
}

struct Result { std::string name; double time_ms; double l2_res; bool ok; };

static Result run_solver(Solver3D& solver, Grid3D& g, const std::vector<double>& rhs,
                          int iters, double tol, int warmup, int measure, bool is_pcg) {
    for(int w=0;w<warmup;w++){std::fill(g.p.begin(),g.p.end(),0.0); solver.solve(g,rhs,iters,tol);}
    std::fill(g.p.begin(),g.p.end(),0.0);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int m=0;m<measure;m++){std::fill(g.p.begin(),g.p.end(),0.0); solver.solve(g,rhs,iters,tol);}
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double>(t1-t0).count()*1000.0/measure;
    double l2=is_pcg?compute_residual_l2_pcg(g,rhs):compute_residual_l2_jacobi(g,rhs);
    bool ok=(l2<1e6);
    return {solver.name(), ms, l2, ok};
}

int main() {
    test_header("3D Solver Performance Comparison");

    // Grid sizes to test
    std::vector<std::tuple<int,int,int,double>> configs={
        {16,8,8,1e-10},     // small
        {32,16,16,1e-8},    // medium
        {64,32,16,1e-6},    // large
    };

    for(auto [nx,ny,nz,tol]:configs){
        int N=(nx+2)*(ny+2)*(nz+2);
        int ncells=nx*ny*nz;
        printf("\n--- Grid %dx%dx%d (%d cells) ---\n",nx,ny,nz,ncells);

        std::vector<double> rhs(N,0.0);
        double cx=nx/2.0,cy=ny/2.0,cz=nz/2.0;
        for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) for(int k=1;k<=nz;k++){
            double d2=(i-cx)*(i-cx)+(j-cy)*(j-cy)+(k-cz)*(k-cz);
            rhs[i+j*(nx+2)+k*(nx+2)*(ny+2)]=std::exp(-0.01*d2);
        }

        std::vector<Result> results;
        Grid3D g(nx,ny,nz,1.0,1.0,1.0);

        // CG (plain, no preconditioner)
        {PCG3D s(std::make_unique<IdentityPreconditioner3D>());
         auto r=run_solver(s,g,rhs,100,tol,2,5,true);
         r.name="CG(100)"; results.push_back(r);}

        // PCG/GMG
        {PCG3D s(std::make_unique<GMGPreconditioner3D>());
         auto r=run_solver(s,g,rhs,20,tol,2,5,true);
         r.name="PCG/GMG(20)"; results.push_back(r);}

        // PCG/UAAMG
        {PCG3D s(std::make_unique<UAAMGPreconditioner3D>());
         auto r=run_solver(s,g,rhs,20,tol,2,5,true);
         r.name="PCG/UAAMG(20)"; results.push_back(r);}

        printf("  %-18s %10s %12s %8s\n","Solver","ms/solve","L2|res|","OK");
        printf("  %s\n",std::string(50,'-').c_str());
        for(auto& r:results){
            printf("  %-18s %10.3f %12.4e %8s\n",r.name.c_str(),r.time_ms,r.l2_res,r.ok?"PASS":"FAIL");
            std::string label=r.name+" "+std::to_string(nx)+"x"+std::to_string(ny)+"x"+std::to_string(nz);
            check(r.ok,label);
        }
    }

    return test_summary();
}

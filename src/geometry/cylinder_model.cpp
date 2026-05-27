#include "geometry/cylinder_model.h"
#include <algorithm>
#include <cmath>

double SmoothCylinder::smoothstep(double t) {
    if (t <= -1.0) return 0.0;
    if (t >=  1.0) return 1.0;
    double s = (t + 1.0) * 0.5;
    return s * s * (3.0 - 2.0 * s);
}

// ═══════════════════ StairStepCylinder ═══════════════════

StairStepCylinder::StairStepCylinder(double cx, double cy, double R,
                                       double dx, double dy)
    : cx_(cx), cy_(cy), R_(R), dx_(dx), dy_(dy) {}

double StairStepCylinder::solidFraction(double x, double y) const {
    double d = (x-cx_)*(x-cx_) + (y-cy_)*(y-cy_);
    return (d < R_*R_) ? 1.0 : 0.0;
}

void StairStepCylinder::applyToLaplacian(Grid& g,
    std::vector<double>& diag, std::vector<double>& ox, std::vector<double>& oy) {
    (void)g; (void)diag; (void)ox; (void)oy;
}

void StairStepCylinder::applyToVelocity(Grid& g) const {
    for (int i=1;i<=g.nx;i++) for(int j=1;j<=g.ny;j++) {
        if (!g.is_solid(i,j)) continue;
        if (i>1   && !g.is_solid(i-1,j)) g.u_at(i-1,j)=0;
        if (i<g.nx && !g.is_solid(i+1,j)) g.u_at(i,j)=0;
        if (j>1   && !g.is_solid(i,j-1)) g.v_at(i,j-1)=0;
        if (j<g.ny && !g.is_solid(i,j+1)) g.v_at(i,j)=0;
    }
}

// ═══════════════════ SmoothCylinder ═══════════════════

SmoothCylinder::SmoothCylinder(double cx, double cy, double R,
                                 double dx, double dy, double tw)
    : cx_(cx), cy_(cy), R_(R), dx_(dx), dy_(dy),
      tw_(tw > 0 ? tw : dx) {}

double SmoothCylinder::solidFraction(double x, double y) const {
    double r = std::sqrt((x-cx_)*(x-cx_) + (y-cy_)*(y-cy_));
    return smoothstep((r - R_) / tw_);
}

void SmoothCylinder::applyToLaplacian(Grid& g,
    std::vector<double>& diag, std::vector<double>& ox, std::vector<double>& oy) {
    const double eps = 0.1, large = 1e15; // min permeability 0.1
    for (int j=1;j<=g.ny;j++) for (int i=1;i<=g.nx;i++) {
        size_t idx = g.ip(i,j);
        double xc=(i-0.5)*g.dx, yc=(j-0.5)*g.dy;
        double phi=solidFraction(xc,yc);
        if (phi>0.9) {
            g.set_solid(i,j);
            diag[idx]=large; ox[idx]=0; oy[idx]=0;
        } else {
            double pL=solidFraction(xc-g.dx,yc), pR=solidFraction(xc+g.dx,yc);
            double pB=solidFraction(xc,yc-g.dy), pT=solidFraction(xc,yc+g.dy);
            double cL=1.0-0.5*(phi+pL)+eps, cR=1.0-0.5*(phi+pR)+eps;
            double cB=1.0-0.5*(phi+pB)+eps, cT=1.0-0.5*(phi+pT)+eps;
            double ix=1.0/(g.dx*g.dx), iy=1.0/(g.dy*g.dy);
            diag[idx]=(cL+cR)*ix+(cB+cT)*iy;
            ox[idx]=-0.5*(cL+cR)*ix;
            oy[idx]=-0.5*(cB+cT)*iy;
        }
    }
}

void SmoothCylinder::applyToVelocity(Grid& g) const {
    for (int i=0;i<=g.nx;i++) for (int j=1;j<=g.ny;j++) {
        double xf=i*g.dx, yf=(j-0.5)*g.dy;
        if (solidFraction(xf,yf)>0.8) g.u_at(i,j)=0;
    }
    for (int i=1;i<=g.nx;i++) for (int j=0;j<=g.ny;j++) {
        double xf=(i-0.5)*g.dx, yf=j*g.dy;
        if (solidFraction(xf,yf)>0.8) g.v_at(i,j)=0;
    }
}

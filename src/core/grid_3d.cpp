#include "core/grid_3d.h"

Grid3D::Grid3D(int nx_, int ny_, int nz_, double lx, double ly, double lz)
    : nx(nx_), ny(ny_), nz(nz_), dx(lx/nx_), dy(ly/ny_), dz(lz/nz_)
{
    int nu = (nx+1) * (ny+2) * (nz+2);
    int nv = (nx+2) * (ny+1) * (nz+2);
    int nw = (nx+2) * (ny+2) * (nz+1);
    int np = (nx+2) * (ny+2) * (nz+2);
    u.assign(nu, 0.0);
    v.assign(nv, 0.0);
    w.assign(nw, 0.0);
    p.assign(np, 0.0);
    solid.assign(np, false);
}

int Grid3D::iu(int i, int j, int k) const { return i + j * (nx+1) + k * (nx+1) * (ny+2); }
int Grid3D::iv(int i, int j, int k) const { return i + j * (nx+2) + k * (nx+2) * (ny+1); }
int Grid3D::iw(int i, int j, int k) const { return i + j * (nx+2) + k * (nx+2) * (ny+2); }
int Grid3D::ip(int i, int j, int k) const { return i + j * (nx+2) + k * (nx+2) * (ny+2); }

double  Grid3D::u_at(int i, int j, int k) const { return u[iu(i,j,k)]; }
double& Grid3D::u_at(int i, int j, int k)       { return u[iu(i,j,k)]; }
double  Grid3D::v_at(int i, int j, int k) const { return v[iv(i,j,k)]; }
double& Grid3D::v_at(int i, int j, int k)       { return v[iv(i,j,k)]; }
double  Grid3D::w_at(int i, int j, int k) const { return w[iw(i,j,k)]; }
double& Grid3D::w_at(int i, int j, int k)       { return w[iw(i,j,k)]; }
double  Grid3D::p_at(int i, int j, int k) const { return p[ip(i,j,k)]; }
double& Grid3D::p_at(int i, int j, int k)       { return p[ip(i,j,k)]; }
bool    Grid3D::is_solid(int i, int j, int k) const { return solid[ip(i,j,k)]; }
void    Grid3D::set_solid(int i, int j, int k)       { solid[ip(i,j,k)] = true; }

double Grid3D::divergence(int i, int j, int k) const {
    return (u_at(i,j,k) - u_at(i-1,j,k)) / dx
         + (v_at(i,j,k) - v_at(i,j-1,k)) / dy
         + (w_at(i,j,k) - w_at(i,j,k-1)) / dz;
}

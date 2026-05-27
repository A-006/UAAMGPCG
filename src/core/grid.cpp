#include "core/grid.h"
#include <cmath>

Grid::Grid(int nx_, int ny_, double lx, double ly)
    : nx(nx_), ny(ny_), dx(lx/nx_), dy(ly/ny_)
{
    int nu = (nx+1) * (ny+2);
    int nv = (nx+2) * (ny+1);
    int np = (nx+2) * (ny+2);
    u.assign(nu, 0.0);
    v.assign(nv, 0.0);
    p.assign(np, 0.0);
    solid.assign(np, false);
}

// Column-major indexing
int Grid::iu(int i, int j) const { return i + j * (nx+1); }
int Grid::iv(int i, int j) const { return i + j * (nx+2); }
int Grid::ip(int i, int j) const { return i + j * (nx+2); }

double  Grid::u_at(int i, int j) const { return u[iu(i,j)]; }
double& Grid::u_at(int i, int j)       { return u[iu(i,j)]; }
double  Grid::v_at(int i, int j) const { return v[iv(i,j)]; }
double& Grid::v_at(int i, int j)       { return v[iv(i,j)]; }
double  Grid::p_at(int i, int j) const { return p[ip(i,j)]; }
double& Grid::p_at(int i, int j)       { return p[ip(i,j)]; }
bool    Grid::is_solid(int i, int j) const { return solid[ip(i,j)]; }
void    Grid::set_solid(int i, int j)       { solid[ip(i,j)] = true; }

void Grid::init_variable_lap() {
    int np = (nx+2) * (ny+2);
    lap_diag.assign(np, 0.0);
    lap_off_x.assign(np, 0.0);
    lap_off_y.assign(np, 0.0);
}

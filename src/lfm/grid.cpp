#include "lfm/grid.h"

Grid::Grid(int nx_, int ny_, double lx, double ly)
    : nx(nx_), ny(ny_), dx(lx/nx_), dy(ly/ny_)
{
    u.resize((nx+1) * (ny+2), 0.0);
    v.resize((nx+2) * (ny+1), 0.0);
    p.resize((nx+2) * (ny+2), 0.0);
    solid.resize((nx+2) * (ny+2), false);
}

int Grid::iu(int i, int j) const { return i*(ny+2) + j; }
double  Grid::u_at(int i, int j) const { return u[iu(i,j)]; }
double& Grid::u_at(int i, int j)       { return u[iu(i,j)]; }

int Grid::iv(int i, int j) const { return i*(ny+1) + j; }
double  Grid::v_at(int i, int j) const { return v[iv(i,j)]; }
double& Grid::v_at(int i, int j)       { return v[iv(i,j)]; }

int Grid::ip(int i, int j) const { return i*(ny+2) + j; }
double  Grid::p_at(int i, int j) const { return p[ip(i,j)]; }
double& Grid::p_at(int i, int j)       { return p[ip(i,j)]; }
bool    Grid::is_solid(int i, int j) const { return solid[ip(i,j)]; }
void    Grid::set_solid(int i, int j)       { solid[ip(i,j)] = true; }

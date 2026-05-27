#include "core/grid.h"

Grid::Grid(int nx_, int ny_, double lx, double ly)
    : Mesh2D(nx_, ny_, lx, ly)
{
    u.assign(u_size(), 0.0);
    v.assign(v_size(), 0.0);
    p.assign(p_size(), 0.0);
    solid.assign(p_size(), false);
}

void Grid::init_variable_lap() {
    int np = p_size();
    lap_diag.assign(np, 0.0);
    lap_off_x.assign(np, 0.0);
    lap_off_y.assign(np, 0.0);
}

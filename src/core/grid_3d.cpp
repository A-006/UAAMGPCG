#include "core/grid_3d.h"

Grid3D::Grid3D(int nx_, int ny_, int nz_, double lx, double ly, double lz)
    : Mesh3D(nx_, ny_, nz_, lx, ly, lz)
{
    u.assign(u_size(), 0.0);
    v.assign(v_size(), 0.0);
    w.assign(w_size(), 0.0);
    p.assign(p_size(), 0.0);
    solid.assign(p_size(), false);
}

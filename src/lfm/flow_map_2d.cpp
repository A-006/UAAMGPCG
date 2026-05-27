#include "lfm/flow_map_2d.h"

FlowMap2D::FlowMap2D(int nx_, int ny_, double dx_, double dy_)
    : nx(nx_), ny(ny_), dx(dx_), dy(dy_)
{
    size_t N = nx * ny;
    phi_x.resize(N); phi_y.resize(N);
    F00.resize(N); F10.resize(N); F01.resize(N); F11.resize(N);
    psi_x.resize(N); psi_y.resize(N);
    T00.resize(N); T10.resize(N); T01.resize(N); T11.resize(N);
}

void FlowMap2D::set_identity() {
    for (int j = 1; j <= ny; j++) {
        for (int i = 1; i <= nx; i++) {
            size_t k = idx(i,j);
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            phi_x[k] = x;  phi_y[k] = y;
            F00[k] = 1.0;  F10[k] = 0.0;
            F01[k] = 0.0;  F11[k] = 1.0;
        }
    }
}

void FlowMap2D::set_backward_identity() {
    for (int j = 1; j <= ny; j++) {
        for (int i = 1; i <= nx; i++) {
            size_t k = idx(i,j);
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            psi_x[k] = x;  psi_y[k] = y;
            T00[k] = 1.0;  T10[k] = 0.0;
            T01[k] = 0.0;  T11[k] = 1.0;
        }
    }
}

#include "integrator/lfm/flow_map_3d.h"

FlowMap3D::FlowMap3D(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_)
    : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_) {
    size_t N = (size_t)nx * ny * nz;
    for (auto* p : {&phi_x, &phi_y, &phi_z, &F00, &F01, &F02, &F10, &F11, &F12, &F20, &F21, &F22,
                    &psi_x, &psi_y, &psi_z, &T00, &T01, &T02, &T10, &T11, &T12, &T20, &T21, &T22})
        p->resize(N);
}

void FlowMap3D::set_identity() {
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                size_t m = idx(i, j, k);
                phi_x[m] = (i - 0.5) * dx;
                phi_y[m] = (j - 0.5) * dy;
                phi_z[m] = (k - 0.5) * dz;
                F00[m] = F11[m] = F22[m] = 1.0;
                F01[m] = F02[m] = F10[m] = F12[m] = F20[m] = F21[m] = 0.0;
            }
}

void FlowMap3D::set_backward_identity() {
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                size_t m = idx(i, j, k);
                psi_x[m] = (i - 0.5) * dx;
                psi_y[m] = (j - 0.5) * dy;
                psi_z[m] = (k - 0.5) * dz;
                T00[m] = T11[m] = T22[m] = 1.0;
                T01[m] = T02[m] = T10[m] = T12[m] = T20[m] = T21[m] = 0.0;
            }
}

#include "boundary/boundary.h"
#include <cmath>

void BoundaryConditions::applyKarman(Grid& g, double U_inf) {
    int nx = g.nx, ny = g.ny;

    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j) = U_inf;
        g.v_at(0, j) = 0.0;
    }
    for (int j = 1; j <= ny; j++) {
        g.u_at(nx, j) = g.u_at(nx-1, j);
        g.v_at(nx+1, j) = g.v_at(nx, j);
    }
    for (int i = 0; i <= nx; i++) {
        g.u_at(i, 0)  = g.u_at(i, 1);
        g.u_at(i, ny+1) = g.u_at(i, ny);
    }
    for (int i = 1; i <= nx; i++) {
        g.v_at(i, 0)  = 0.0;
        g.v_at(i, ny) = 0.0;
    }
    g.v_at(0, 0) = g.v_at(0, ny) = 0.0;
    g.v_at(nx+1, 0) = g.v_at(nx+1, ny) = 0.0;
}

void BoundaryConditions::applySmoke(Grid& g) {
    int nx = g.nx, ny = g.ny;

    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j) = 0.0;
        g.u_at(nx, j) = 0.0;
    }
    for (int i = 1; i <= nx; i++) {
        g.v_at(i, 0) = 0.0;
        g.v_at(i, ny) = 0.0;
    }
    for (int i = 0; i <= nx; i++) {
        g.u_at(i, 0) = -g.u_at(i, 1);
        g.u_at(i, ny+1) = -g.u_at(i, ny);
    }
    for (int j = 0; j <= ny; j++) {
        g.v_at(0, j) = -g.v_at(1, j);
        g.v_at(nx+1, j) = -g.v_at(nx, j);
    }
}

void BoundaryConditions::applySolid(Grid& g) {
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            if (!g.is_solid(i,j)) continue;
            if (i > 1 && !g.is_solid(i-1,j))       g.u_at(i-1, j) = 0.0;
            if (i < g.nx && !g.is_solid(i+1,j))    g.u_at(i, j)   = 0.0;
            if (j > 1 && !g.is_solid(i, j-1))      g.v_at(i, j-1) = 0.0;
            if (j < g.ny && !g.is_solid(i, j+1))   g.v_at(i, j)   = 0.0;
        }
    }
}

void BoundaryConditions::setupCylinder(Grid& g, double cx, double cy, double R) {
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            double xc = (i - 0.5) * g.dx;
            double yc = (j - 0.5) * g.dy;
            if ((xc-cx)*(xc-cx) + (yc-cy)*(yc-cy) < R*R)
                g.set_solid(i, j);
        }
    }
}

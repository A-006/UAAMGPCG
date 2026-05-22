#include "lfm/vtk_io.h"
#include "lfm/advection.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

void write_vtk(const Grid& g, int frame, const Config& cfg) {
    std::ostringstream ss;
    ss << cfg.out_dir << "/frame_" << std::setw(5) << std::setfill('0') << frame << ".vtk";
    std::ofstream f(ss.str());
    if (!f) { std::cerr << "Cannot write " << ss.str() << "\n"; return; }

    f << std::scientific << std::setprecision(6);

    f << "# vtk DataFile Version 2.0\n";
    f << "LFM 2D Fluid - Frame " << frame << "\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << (g.nx+1) << " " << (g.ny+1) << " 1\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << g.dx << " " << g.dy << " 1\n";

    int np = (g.nx+1) * (g.ny+1);
    f << "POINT_DATA " << np << "\n";

    // Velocity (interpolated at nodes)
    f << "VECTORS velocity float\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            double u_sum = 0.0, v_sum = 0.0;
            int nu = 0, nv = 0;

            if (i > 0 && j > 0 && j <= g.ny) { u_sum += g.u_at(i-1, j); nu++; }
            if (i < g.nx && j > 0 && j <= g.ny) { u_sum += g.u_at(i, j); nu++; }
            if (j > 0 && i > 0 && i <= g.nx) { v_sum += g.v_at(i, j-1); nv++; }
            if (j < g.ny && i > 0 && i <= g.nx) { v_sum += g.v_at(i, j); nv++; }

            f << ((nu>0) ? u_sum/nu : 0.0) << " "
              << ((nv>0) ? v_sum/nv : 0.0) << " 0\n";
        }
    }

    // Vorticity
    f << "SCALARS vorticity float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            double dvdx = 0.0, dudy = 0.0;
            if (i > 0 && i < g.nx && j > 0 && j <= g.ny)
                dvdx = (g.v_at(i+1,j) - g.v_at(i,j)) / g.dx;
            if (i > 0 && i <= g.nx && j > 0 && j < g.ny)
                dudy = (g.u_at(i,j+1) - g.u_at(i,j)) / g.dy;
            f << (dvdx - dudy) << "\n";
        }
    }

    // Divergence
    f << "SCALARS divergence float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            int ci = clamp(i, 1, g.nx);
            int cj = clamp(j, 1, g.ny);
            f << divergence(g, ci, cj) << "\n";
        }
    }

    // Solid marker
    f << "SCALARS solid float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            int ci = clamp(i, 1, g.nx);
            int cj = clamp(j, 1, g.ny);
            f << (g.is_solid(ci,cj) ? 1.0 : 0.0) << "\n";
        }
    }

    f.close();
}

void print_status(int step, double t, const Grid& g) {
    double max_u = 0.0, max_div = 0.0;
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j)) continue;
            double uc = 0.5 * (g.u_at(i-1,j) + g.u_at(i,j));
            double vc = 0.5 * (g.v_at(i,j-1) + g.v_at(i,j));
            max_u = std::max(max_u, std::sqrt(uc*uc + vc*vc));
            max_div = std::max(max_div, std::abs(divergence(g,i,j)));
        }
    }
    std::cout << "  step=" << std::setw(6) << step
              << "  t=" << std::fixed << std::setprecision(4) << t
              << "  max|u|=" << std::setprecision(3) << max_u
              << "  max|div|=" << std::scientific << std::setprecision(3) << max_div
              << std::fixed << "\n";
}

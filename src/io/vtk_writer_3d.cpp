#include "io/vtk_writer_3d.h"
#include "ops/operators_3d.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

void VtkWriter3D::write(const Grid3D& g, int frame, const Config& cfg) {
    std::ostringstream ss;
    ss << cfg.out_dir << "/frame_" << std::setw(5) << std::setfill('0') << frame << ".vtk";
    std::ofstream f(ss.str());
    if (!f) { std::cerr << "Cannot write " << ss.str() << "\n"; return; }

    f << std::scientific << std::setprecision(6);
    f << "# vtk DataFile Version 2.0\n";
    f << "LFM 3D Fluid - Frame " << frame << "\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << (g.nx + 1) << " " << (g.ny + 1) << " " << (g.nz + 1) << "\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << g.dx << " " << g.dy << " " << g.dz << "\n";

    int np = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    f << "POINT_DATA " << np << "\n";

    // Velocity at nodes (interpolated from MAC faces)
    f << "VECTORS velocity float\n";
    for (int k = 0; k <= g.nz; k++) {
        for (int j = 0; j <= g.ny; j++) {
            for (int i = 0; i <= g.nx; i++) {
                double u_sum = 0.0, v_sum = 0.0, w_sum = 0.0;
                int nu = 0, nv = 0, nw = 0;
                if (i > 0 && j > 0 && j <= g.ny && k > 0 && k <= g.nz)
                    { u_sum += g.u_at(i - 1, j, k); nu++; }
                if (i < g.nx && j > 0 && j <= g.ny && k > 0 && k <= g.nz)
                    { u_sum += g.u_at(i, j, k); nu++; }
                if (j > 0 && i > 0 && i <= g.nx && k > 0 && k <= g.nz)
                    { v_sum += g.v_at(i, j - 1, k); nv++; }
                if (j < g.ny && i > 0 && i <= g.nx && k > 0 && k <= g.nz)
                    { v_sum += g.v_at(i, j, k); nv++; }
                if (k > 0 && i > 0 && i <= g.nx && j > 0 && j <= g.ny)
                    { w_sum += g.w_at(i, j, k - 1); nw++; }
                if (k < g.nz && i > 0 && i <= g.nx && j > 0 && j <= g.ny)
                    { w_sum += g.w_at(i, j, k); nw++; }
                f << ((nu > 0) ? u_sum / nu : 0.0) << " "
                  << ((nv > 0) ? v_sum / nv : 0.0) << " "
                  << ((nw > 0) ? w_sum / nw : 0.0) << "\n";
            }
        }
    }

    // Vorticity magnitude — useful for iso-surface visualization in ParaView.
    f << "SCALARS vorticity_magnitude float 1\nLOOKUP_TABLE default\n";
    for (int k = 0; k <= g.nz; k++) {
        for (int j = 0; j <= g.ny; j++) {
            for (int i = 0; i <= g.nx; i++) {
                int ci = Mesh3D::clamp(i, 1, g.nx);
                int cj = Mesh3D::clamp(j, 1, g.ny);
                int ck = Mesh3D::clamp(k, 1, g.nz);
                f << fvc::vorticity_magnitude(g, ci, cj, ck) << "\n";
            }
        }
    }

    // Divergence — diagnostic, should stay near zero post-projection
    f << "SCALARS divergence float 1\nLOOKUP_TABLE default\n";
    for (int k = 0; k <= g.nz; k++) {
        for (int j = 0; j <= g.ny; j++) {
            for (int i = 0; i <= g.nx; i++) {
                int ci = Mesh3D::clamp(i, 1, g.nx);
                int cj = Mesh3D::clamp(j, 1, g.ny);
                int ck = Mesh3D::clamp(k, 1, g.nz);
                f << g.divergence(ci, cj, ck) << "\n";
            }
        }
    }

    f << "SCALARS solid float 1\nLOOKUP_TABLE default\n";
    for (int k = 0; k <= g.nz; k++) {
        for (int j = 0; j <= g.ny; j++) {
            for (int i = 0; i <= g.nx; i++) {
                int ci = Mesh3D::clamp(i, 1, g.nx);
                int cj = Mesh3D::clamp(j, 1, g.ny);
                int ck = Mesh3D::clamp(k, 1, g.nz);
                f << (g.is_solid(ci, cj, ck) ? 1.0 : 0.0) << "\n";
            }
        }
    }
    f.close();
}

void VtkWriter3D::printStatus(int step, double t, const Grid3D& g) {
    double max_u = 0.0, max_div = 0.0, max_omega = 0.0;
    for (int k = 1; k <= g.nz; k++) {
        for (int j = 1; j <= g.ny; j++) {
            for (int i = 1; i <= g.nx; i++) {
                if (g.is_solid(i, j, k)) continue;
                double uc = 0.5 * (g.u_at(i - 1, j, k) + g.u_at(i, j, k));
                double vc = 0.5 * (g.v_at(i, j - 1, k) + g.v_at(i, j, k));
                double wc = 0.5 * (g.w_at(i, j, k - 1) + g.w_at(i, j, k));
                max_u = std::max(max_u, std::sqrt(uc * uc + vc * vc + wc * wc));
                max_div = std::max(max_div, std::abs(g.divergence(i, j, k)));
                max_omega = std::max(max_omega, fvc::vorticity_magnitude(g, i, j, k));
            }
        }
    }
    std::cout << "  step=" << std::setw(6) << step
              << "  t=" << std::fixed << std::setprecision(4) << t
              << "  |u|max=" << std::setprecision(3) << max_u
              << "  |ω|max=" << std::setprecision(3) << max_omega
              << "  |div|max=" << std::scientific << std::setprecision(2) << max_div
              << std::fixed << "\n";
}

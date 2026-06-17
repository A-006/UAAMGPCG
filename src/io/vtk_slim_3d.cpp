#include "io/vtk_slim_3d.h"
#include "numerics/ops/operators_3d.h"
#include <cstdio>
#include <fstream>
#include <vector>

namespace io3d {

void write_vort_vtk(const Grid3D& g, int frame, const std::string& dir) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/frame_%05d.vtk", dir.c_str(), frame);
    std::ofstream f(path);
    f << "# vtk DataFile Version 2.0\nLFM 3D vorticity - Frame " << frame
      << "\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << g.nx + 1 << " " << g.ny + 1 << " " << g.nz + 1 << "\n";
    f << "ORIGIN 0 0 0\nSPACING " << g.dx << " " << g.dy << " " << g.dz << "\n";
    long npts = (long)(g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    f << "POINT_DATA " << npts << "\nSCALARS vorticity_magnitude float 1\nLOOKUP_TABLE default\n";
    for (int k = 0; k <= g.nz; k++)
        for (int j = 0; j <= g.ny; j++)
            for (int i = 0; i <= g.nx; i++) {
                int ci = i < 1 ? 1 : (i > g.nx ? g.nx : i);
                int cj = j < 1 ? 1 : (j > g.ny ? g.ny : j);
                int ck = k < 1 ? 1 : (k > g.nz ? g.nz : k);
                f << (float)fvc::vorticity_magnitude(g, ci, cj, ck) << "\n";
            }
}

void write_vel_raw(const Grid3D& g, int frame, const std::string& dir) {
    long n = (long)g.nx * g.ny * g.nz;
    std::vector<float> ux(n), uy(n), uz(n);
    long c = 0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            for (int k = 1; k <= g.nz; k++) {
                ux[c] = (float)(0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k)));
                uy[c] = (float)(0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k)));
                uz[c] = (float)(0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1)));
                c++;
            }
    char p[512];
    auto wr = [&](const char* nm, const std::vector<float>& b) {
        std::snprintf(p, sizeof(p), "%s/%s_%05d.raw", dir.c_str(), nm, frame);
        std::ofstream f(p, std::ios::binary);
        f.write((const char*)b.data(), b.size() * sizeof(float));
    };
    wr("vx", ux);
    wr("vy", uy);
    wr("vz", uz);
}

void write_face_ic(const Grid3D& g, const std::string& dir) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    char p[512];
    auto wr = [&](const char* nm, const std::vector<float>& b) {
        std::snprintf(p, sizeof(p), "%s/%s.raw", dir.c_str(), nm);
        std::ofstream f(p, std::ios::binary);
        f.write((const char*)b.data(), b.size() * sizeof(float));
    };
    {   // x-faces: (nx+1, ny, nz), a[ix,iy,iz] = u_at(ix, iy+1, iz+1)
        std::vector<float> b((long)(nx + 1) * ny * nz);
        long c = 0;
        for (int ix = 0; ix <= nx; ix++)
            for (int iy = 0; iy < ny; iy++)
                for (int iz = 0; iz < nz; iz++)
                    b[c++] = (float)g.u_at(ix, iy + 1, iz + 1);
        wr("icx", b);
    }
    {   // y-faces: (nx, ny+1, nz), a[ix,iy,iz] = v_at(ix+1, iy, iz+1)
        std::vector<float> b((long)nx * (ny + 1) * nz);
        long c = 0;
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy <= ny; iy++)
                for (int iz = 0; iz < nz; iz++)
                    b[c++] = (float)g.v_at(ix + 1, iy, iz + 1);
        wr("icy", b);
    }
    {   // z-faces: (nx, ny, nz+1), a[ix,iy,iz] = w_at(ix+1, iy+1, iz)
        std::vector<float> b((long)nx * ny * (nz + 1));
        long c = 0;
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy < ny; iy++)
                for (int iz = 0; iz <= nz; iz++)
                    b[c++] = (float)g.w_at(ix + 1, iy + 1, iz);
        wr("icz", b);
    }
}

} // namespace io3d

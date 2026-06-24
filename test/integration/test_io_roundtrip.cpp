// IO module round-trip tests:
//   - io3d::write_face_ic  ->  read back ic{x,y,z}.raw and verify layout/values
//   - load_raw_ic-equivalent reload into a fresh Grid3D
//   - io3d::write_vort_vtk  ->  parse header (DIMENSIONS/SPACING) and count points
//   - 2D VtkWriter::write   ->  parse DIMENSIONS + first velocity/vorticity values
//
// Uses ONLY real headers under include/.
#include "../test_utils.h"

#include "io/vtk_slim_3d.h"
#include "io/vtk_writer.h"
#include "mesh/grid_3d.h"
#include "mesh/grid.h"
#include "core/config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

// Distinct encoding per face so any mis-indexing shows up immediately.
static double enc_u(int ix, int iy, int iz) { return 1.0e6 + 1.0 * ix + 1.0e3 * iy + 1.0e0 * 0 + 7.0 * iz + 0.5; }
static double enc_v(int ix, int iy, int iz) { return 2.0e6 + 11.0 * ix + 1.0e3 * iy + 3.0 * iz + 0.25; }
static double enc_w(int ix, int iy, int iz) { return 3.0e6 + 13.0 * ix + 17.0 * iy + 1.0e3 * iz + 0.75; }

// Read a headerless float32 file into a vector.
static std::vector<float> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read((char*)buf.data(), bytes);
    return buf;
}

int main() {
    test_header("IO round-trip (vtk_slim_3d / vtk_writer)");

    // Unique temp dir.
    std::string dir = "/tmp/iotest_" + std::to_string((long)getpid());
    std::string mk = "mkdir -p " + dir;
    int rc = std::system(mk.c_str());
    check(rc == 0, "created temp dir " + dir);

    // ── Build a Grid3D and stamp every relevant MAC face with a unique value ──
    const int nx = 4, ny = 3, nz = 5;
    Grid3D g(nx, ny, nz, 1.0, 1.0, 1.0);

    for (int ix = 0; ix <= nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz < nz; iz++)
                g.u_at(ix, iy + 1, iz + 1) = enc_u(ix, iy, iz);
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy <= ny; iy++)
            for (int iz = 0; iz < nz; iz++)
                g.v_at(ix + 1, iy, iz + 1) = enc_v(ix, iy, iz);
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz <= nz; iz++)
                g.w_at(ix + 1, iy + 1, iz) = enc_w(ix, iy, iz);

    // ── write_face_ic ──
    io3d::write_face_ic(g, dir);

    auto bx = read_raw(dir + "/icx.raw");
    auto by = read_raw(dir + "/icy.raw");
    auto bz = read_raw(dir + "/icz.raw");

    const long sx = (long)(nx + 1) * ny * nz;
    const long sy = (long)nx * (ny + 1) * nz;
    const long sz = (long)nx * ny * (nz + 1);

    check((long)bx.size() == sx, "icx.raw size == (nx+1)*ny*nz");
    check((long)by.size() == sy, "icy.raw size == nx*(ny+1)*nz");
    check((long)bz.size() == sz, "icz.raw size == nx*ny*(nz+1)");

    // Verify C-order (iz fastest, then iy, then ix) matching the writer loop.
    bool x_ok = true;
    {
        long c = 0;
        for (int ix = 0; ix <= nx && x_ok; ix++)
            for (int iy = 0; iy < ny && x_ok; iy++)
                for (int iz = 0; iz < nz; iz++) {
                    if ((double)bx[c++] != (float)enc_u(ix, iy, iz)) { x_ok = false; break; }
                }
    }
    check(x_ok, "icx values match u_at(ix,iy+1,iz+1) in C-order (iz fastest)");

    bool y_ok = true;
    {
        long c = 0;
        for (int ix = 0; ix < nx && y_ok; ix++)
            for (int iy = 0; iy <= ny && y_ok; iy++)
                for (int iz = 0; iz < nz; iz++) {
                    if ((double)by[c++] != (float)enc_v(ix, iy, iz)) { y_ok = false; break; }
                }
    }
    check(y_ok, "icy values match v_at(ix+1,iy,iz+1) in C-order (iz fastest)");

    bool z_ok = true;
    {
        long c = 0;
        for (int ix = 0; ix < nx && z_ok; ix++)
            for (int iy = 0; iy < ny && z_ok; iy++)
                for (int iz = 0; iz <= nz; iz++) {
                    if ((double)bz[c++] != (float)enc_w(ix, iy, iz)) { z_ok = false; break; }
                }
    }
    check(z_ok, "icz values match w_at(ix+1,iy+1,iz) in C-order (iz fastest)");

    // Spot-check explicit C-order index math for icx: a[ix,iy,iz] at iz + nz*iy + nz*ny*ix.
    {
        int ix = 2, iy = 1, iz = 3;
        long idx = (long)iz + (long)nz * iy + (long)nz * ny * ix;
        check((double)bx[idx] == (float)enc_u(ix, iy, iz),
              "icx C-order index iz+nz*iy+nz*ny*ix resolves to correct face");
    }

    // ── Reload into a fresh Grid3D using the documented load_raw_ic layout ──
    // (load_raw_ic is file-static in scene_3d.cpp; replicate its exact loop.)
    Grid3D g2(nx, ny, nz, 1.0, 1.0, 1.0);
    {
        long c = 0;
        for (int ix = 0; ix <= nx; ix++)
            for (int iy = 0; iy < ny; iy++)
                for (int iz = 0; iz < nz; iz++)
                    g2.u_at(ix, iy + 1, iz + 1) = bx[c++];
        c = 0;
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy <= ny; iy++)
                for (int iz = 0; iz < nz; iz++)
                    g2.v_at(ix + 1, iy, iz + 1) = by[c++];
        c = 0;
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy < ny; iy++)
                for (int iz = 0; iz <= nz; iz++)
                    g2.w_at(ix + 1, iy + 1, iz) = bz[c++];
    }
    bool reload_u_ok = true, reload_v_ok = true, reload_w_ok = true;
    for (int ix = 0; ix <= nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz < nz; iz++)
                if (g2.u_at(ix, iy + 1, iz + 1) != (float)g.u_at(ix, iy + 1, iz + 1)) reload_u_ok = false;
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy <= ny; iy++)
            for (int iz = 0; iz < nz; iz++)
                if (g2.v_at(ix + 1, iy, iz + 1) != (float)g.v_at(ix + 1, iy, iz + 1)) reload_v_ok = false;
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            for (int iz = 0; iz <= nz; iz++)
                if (g2.w_at(ix + 1, iy + 1, iz) != (float)g.w_at(ix + 1, iy + 1, iz)) reload_w_ok = false;
    check(reload_u_ok, "reloaded u-faces equal original (float32 cast)");
    check(reload_v_ok, "reloaded v-faces equal original (float32 cast)");
    check(reload_w_ok, "reloaded w-faces equal original (float32 cast)");

    // ── write_vort_vtk: parse header + count points ──
    io3d::write_vort_vtk(g, 7, dir);
    char vp[512];
    std::snprintf(vp, sizeof(vp), "%s/frame_%05d.vtk", dir.c_str(), 7);
    {
        std::ifstream vf(vp);
        check((bool)vf, "write_vort_vtk produced frame_00007.vtk");
        std::string line;
        int dimx = -1, dimy = -1, dimz = -1;
        long declared_pts = -1;
        long scalar_count = 0;
        bool in_scalars = false;
        while (std::getline(vf, line)) {
            std::istringstream ls(line);
            std::string tok;
            ls >> tok;
            if (tok == "DIMENSIONS") {
                ls >> dimx >> dimy >> dimz;
            } else if (tok == "POINT_DATA") {
                ls >> declared_pts;
            } else if (tok == "LOOKUP_TABLE") {
                in_scalars = true;
            } else if (in_scalars && !tok.empty()) {
                // each remaining non-keyword line is one scalar value
                scalar_count++;
            }
        }
        check(dimx == nx + 1 && dimy == ny + 1 && dimz == nz + 1,
              "vort vtk DIMENSIONS == (nx+1,ny+1,nz+1)");
        long expect_pts = (long)(nx + 1) * (ny + 1) * (nz + 1);
        check(declared_pts == expect_pts, "vort vtk POINT_DATA == (nx+1)(ny+1)(nz+1)");
        check(scalar_count == expect_pts, "vort vtk wrote one scalar per point");
    }

    // ── 2D VtkWriter::write: parse DIMENSIONS + a velocity value ──
    {
        const int n2x = 4, n2y = 2;
        Grid g2d(n2x, n2y, 4.0, 1.0);
        // Set a known constant u everywhere; node velocity (interior) should average to it.
        for (size_t i = 0; i < g2d.u.size(); i++) g2d.u[i] = 3.0;
        for (size_t i = 0; i < g2d.v.size(); i++) g2d.v[i] = 0.0;

        Config cfg;
        cfg.out_dir = dir;
        VtkWriter::write(g2d, 2, cfg);

        char p2[512];
        std::snprintf(p2, sizeof(p2), "%s/frame_%05d.vtk", dir.c_str(), 2);
        std::ifstream f2(p2);
        check((bool)f2, "2D VtkWriter produced frame_00002.vtk");

        std::string line;
        int dimx = -1, dimy = -1, dimz = -1;
        long declared_pts = -1;
        std::vector<std::string> vel_lines;
        std::string section;
        bool reading_vel = false;
        while (std::getline(f2, line)) {
            std::istringstream ls(line);
            std::string tok;
            ls >> tok;
            if (tok == "DIMENSIONS") {
                ls >> dimx >> dimy >> dimz;
            } else if (tok == "POINT_DATA") {
                ls >> declared_pts;
            } else if (tok == "VECTORS") {
                reading_vel = true;
                continue;
            } else if (tok == "SCALARS") {
                reading_vel = false;
            } else if (reading_vel && !tok.empty()) {
                vel_lines.push_back(line);
            }
        }
        check(dimx == n2x + 1 && dimy == n2y + 1 && dimz == 1,
              "2D vtk DIMENSIONS == (nx+1,ny+1,1)");
        check(declared_pts == (long)(n2x + 1) * (n2y + 1),
              "2D vtk POINT_DATA == (nx+1)(ny+1)");
        check((long)vel_lines.size() == (long)(n2x + 1) * (n2y + 1),
              "2D vtk wrote one velocity vector per point");

        // Interior node (i in [1,nx-1], j in [1,ny]) averages two u==3 faces -> 3.
        // Node index for (i,j) in writer's j-outer/i-inner loop: j*(nx+1)+i.
        if (!vel_lines.empty()) {
            int i = 2, j = 1;
            long node = (long)j * (n2x + 1) + i;
            std::istringstream vs(vel_lines[node]);
            double ux = -999, uy = -999, uz = -999;
            vs >> ux >> uy >> uz;
            check_approx(ux, 3.0, 1e-9, "2D vtk interior node velocity_x averages to u=3");
            check_approx(uy, 0.0, 1e-9, "2D vtk interior node velocity_y == 0");
            check_approx(uz, 0.0, 1e-12, "2D vtk velocity_z component == 0");
        }
    }

    // Cleanup (best effort).
    std::string rm = "rm -rf " + dir;
    std::system(rm.c_str());

    return test_summary();
}

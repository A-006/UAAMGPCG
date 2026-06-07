/**
 * @file dump_collision_ic.cpp
 * @brief Generate the shared two-ring collision initial condition and dump it
 *        in the authors' MAC layout, so BOTH our LFM and the authors' headless
 *        LFM start from the IDENTICAL velocity field (cross-validation).
 *
 * Uses the exact same scenarios::add_vortex_ring setup as run_collision_paper,
 * then writes three raw float32 files in C-order:
 *   icx.raw  shape (nx+1, ny, nz)   = u on x-faces  (author init_u_x)
 *   icy.raw  shape (nx, ny+1, nz)   = v on y-faces  (author init_u_y)
 *   icz.raw  shape (nx, ny, nz+1)   = w on z-faces  (author init_u_z)
 * A companion python step wraps these into init_u_{x,y,z}.npy.
 *
 * Usage: dump_collision_ic [NX] [out_dir]   defaults: NX=128 out=/tmp/coll_ic
 */
#include "core/grid_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>

int main(int argc, char** argv) {
    int NX            = argc > 1 ? std::atoi(argv[1]) : 128;
    std::string outdir = argc > 2 ? argv[2] : "/tmp/coll_ic";
    int NY = 2 * NX, NZ = 2 * NX;     // collision axis x is the short one (paper aspect)
    double Lx = 0.5, Ly = 1.0, Lz = 1.0; // dx=dy=dz uniform
    mkdir(outdir.c_str(), 0755);

    Grid3D g(NX, NY, NZ, Lx, Ly, Lz);

    // EXACT same rings as run_collision_paper.cu.
    scenarios::VortexRing left;
    left.center      = {0.35 * Lx, 0.5 * Ly, 0.5 * Lz};
    left.axis        = {1.0, 0.0, 0.0};
    left.radius      = 0.10;
    left.core        = 0.022;
    left.circulation = +1.0;
    left.n_segments  = 300;
    scenarios::VortexRing right = left;
    right.center                = {0.65 * Lx, 0.5 * Ly, 0.5 * Lz};
    right.circulation           = -1.0;
    scenarios::add_vortex_ring(g, left);
    scenarios::add_vortex_ring(g, right);

    // x-faces: (nx+1, ny, nz), a[ix,iy,iz] = u_at(ix, iy+1, iz+1)
    {
        std::vector<float> buf((long)(NX + 1) * NY * NZ);
        long c = 0;
        for (int ix = 0; ix <= NX; ix++)
            for (int iy = 0; iy < NY; iy++)
                for (int iz = 0; iz < NZ; iz++)
                    buf[c++] = (float)g.u_at(ix, iy + 1, iz + 1);
        std::ofstream f(outdir + "/icx.raw", std::ios::binary);
        f.write((char*)buf.data(), buf.size() * sizeof(float));
    }
    // y-faces: (nx, ny+1, nz), a[ix,iy,iz] = v_at(ix+1, iy, iz+1)
    {
        std::vector<float> buf((long)NX * (NY + 1) * NZ);
        long c = 0;
        for (int ix = 0; ix < NX; ix++)
            for (int iy = 0; iy <= NY; iy++)
                for (int iz = 0; iz < NZ; iz++)
                    buf[c++] = (float)g.v_at(ix + 1, iy, iz + 1);
        std::ofstream f(outdir + "/icy.raw", std::ios::binary);
        f.write((char*)buf.data(), buf.size() * sizeof(float));
    }
    // z-faces: (nx, ny, nz+1), a[ix,iy,iz] = w_at(ix+1, iy+1, iz)
    {
        std::vector<float> buf((long)NX * NY * (NZ + 1));
        long c = 0;
        for (int ix = 0; ix < NX; ix++)
            for (int iy = 0; iy < NY; iy++)
                for (int iz = 0; iz <= NZ; iz++)
                    buf[c++] = (float)g.w_at(ix + 1, iy + 1, iz);
        std::ofstream f(outdir + "/icz.raw", std::ios::binary);
        f.write((char*)buf.data(), buf.size() * sizeof(float));
    }
    std::printf("dumped collision IC (%dx%dx%d) to %s/ic{x,y,z}.raw\n", NX, NY, NZ, outdir.c_str());
    return 0;
}

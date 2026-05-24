#pragma once
#include <vector>
#include <cstddef>

/// 3D MAC grid — skeleton for future extension.
struct Grid3D {
    int nx, ny, nz;
    double dx, dy, dz;
    std::vector<double> u, v, w, p;
    std::vector<bool> solid;

    Grid3D(int nx_, int ny_, int nz_, double lx, double ly, double lz);

    int iu(int i, int j, int k) const;
    int iv(int i, int j, int k) const;
    int iw(int i, int j, int k) const;
    int ip(int i, int j, int k) const;

    double  u_at(int i, int j, int k) const;
    double& u_at(int i, int j, int k);
    double  v_at(int i, int j, int k) const;
    double& v_at(int i, int j, int k);
    double  w_at(int i, int j, int k) const;
    double& w_at(int i, int j, int k);
    double  p_at(int i, int j, int k) const;
    double& p_at(int i, int j, int k);
    bool    is_solid(int i, int j, int k) const;
    void    set_solid(int i, int j, int k);

    double divergence(int i, int j, int k) const;

    static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
};

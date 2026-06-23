// ─────────────────────────────────────────────────────────────────────────
// Semi-structured adaptive grid (3D) — octree analogue of adaptive_grid_2d.h.
// A stack of uniform n_l³ level grids (n_l = n0*2^l, h_l = 1/n_l) plus a
// leaf/refined/outside mask. Cells: OUTSIDE / LEAF (active DOF) / REFINED
// (parent of 8 finer cells). The union of LEAF cells tiles [0,1]³ once. Built
// top-down from a refinement predicate, then 2:1 graded over the 6 face
// neighbours. Reuses the Cell / BC enums from the 2D header.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include "adaptive_grid_2d.h"  // Cell, BC enums
#include <vector>
#include <cstdint>
#include <functional>
#include <cassert>
#include <array>

namespace semistruct {

struct Level3D {
    int n = 0;
    double h = 0.0;
    std::vector<Cell> type;   // n*n*n, index i + j*n + k*n*n
    std::vector<int> dof;     // global DOF id for LEAF cells, else -1

    inline int idx(int i, int j, int k) const { return i + j * n + k * n * n; }
    inline bool in(int i, int j, int k) const {
        return i >= 0 && i < n && j >= 0 && j < n && k >= 0 && k < n;
    }
    inline Cell at(int i, int j, int k) const { return type[idx(i, j, k)]; }
};

struct AdaptiveGrid3D {
    int L = 0;
    int n0 = 0;
    std::vector<Level3D> lev;
    int ndof = 0;
    std::vector<int> dof_level, dof_i, dof_j, dof_k;
    std::array<BC, 6> bc{BC::Neumann, BC::Neumann, BC::Neumann,
                         BC::Neumann, BC::Neumann, BC::Neumann};  // -x,+x,-y,+y,-z,+z

    struct DirFace { double kappa; int side; };

    // refineFn(level, i, j, k, h, cx, cy, cz) → true to split into 8 children.
    using RefineFn = std::function<bool(int, int, int, int, double, double, double, double)>;

    void build(int levels, int base_n, const RefineFn& refineFn) {
        L = levels;
        n0 = base_n;
        lev.assign(L, Level3D{});
        for (int l = 0; l < L; ++l) {
            int n = n0 << l;
            lev[l].n = n;
            lev[l].h = 1.0 / n;
            lev[l].type.assign((size_t)n * n * n, Cell::OUTSIDE);
        }
        for (int k = 0; k < n0; ++k)
            for (int j = 0; j < n0; ++j)
                for (int i = 0; i < n0; ++i)
                    mark(0, i, j, k, refineFn);
        enforceGrading();
        numberDofs();
    }

    inline double cx(int l, int i) const { return (i + 0.5) * lev[l].h; }
    inline double cy(int l, int j) const { return (j + 0.5) * lev[l].h; }
    inline double cz(int l, int k) const { return (k + 0.5) * lev[l].h; }

    // Covering leaf of cell (l,i,j,k): walk up to a LEAF ancestor. -1 if REFINED.
    int coveringLeaf(int l, int i, int j, int k, int& ii, int& jj, int& kk) const {
        int cl = l, ci = i, cj = j, ck = k;
        while (cl >= 0) {
            Cell c = lev[cl].at(ci, cj, ck);
            if (c == Cell::LEAF) { ii = ci; jj = cj; kk = ck; return cl; }
            if (c == Cell::REFINED) return -1;
            cl -= 1; ci >>= 1; cj >>= 1; ck >>= 1;
        }
        return -1;
    }

    // The 4 children of refined neighbour (ni,nj,nk) at level+1 touching face s of
    // the COARSE cell (s: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z). Children lie on the
    // neighbour's side facing the coarse cell.
    static void childrenOnFace(int ni, int nj, int nk, int s, int cs[4][3]) {
        int bi = 2 * ni, bj = 2 * nj, bk = 2 * nk;
        int t = 0;
        // fixed coordinate along the face normal; the other two range over {0,1}.
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                int x = bi, y = bj, z = bk;
                switch (s) {
                    case 0: x = bi + 1; y = bj + a; z = bk + b; break;  // coarse -x → neighbour +x
                    case 1: x = bi;     y = bj + a; z = bk + b; break;  // coarse +x → neighbour -x
                    case 2: x = bi + a; y = bj + 1; z = bk + b; break;  // coarse -y → neighbour +y
                    case 3: x = bi + a; y = bj;     z = bk + b; break;  // coarse +y → neighbour -y
                    case 4: x = bi + a; y = bj + b; z = bk + 1; break;  // coarse -z → neighbour +z
                    default: x = bi + a; y = bj + b; z = bk;    break;  // coarse +z → neighbour -z
                }
                cs[t][0] = x; cs[t][1] = y; cs[t][2] = z; ++t;
            }
    }

    inline int dofAt(int l, int i, int j, int k) const { return lev[l].dof[lev[l].idx(i, j, k)]; }

    static constexpr int FDI[6] = {-1, 1, 0, 0, 0, 0};
    static constexpr int FDJ[6] = {0, 0, -1, 1, 0, 0};
    static constexpr int FDK[6] = {0, 0, 0, 0, -1, 1};

private:
    void mark(int l, int i, int j, int k, const RefineFn& refineFn) {
        if (l == L - 1) { lev[l].type[lev[l].idx(i, j, k)] = Cell::LEAF; return; }
        double h = lev[l].h;
        if (refineFn(l, i, j, k, h, cx(l, i), cy(l, j), cz(l, k))) {
            lev[l].type[lev[l].idx(i, j, k)] = Cell::REFINED;
            for (int dk = 0; dk < 2; ++dk)
                for (int dj = 0; dj < 2; ++dj)
                    for (int di = 0; di < 2; ++di)
                        mark(l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk, refineFn);
        } else {
            lev[l].type[lev[l].idx(i, j, k)] = Cell::LEAF;
        }
    }

    void splitLeaf(int l, int i, int j, int k) {
        lev[l].type[lev[l].idx(i, j, k)] = Cell::REFINED;
        for (int dk = 0; dk < 2; ++dk)
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di)
                    lev[l + 1].type[lev[l + 1].idx(2 * i + di, 2 * j + dj, 2 * k + dk)] = Cell::LEAF;
    }

    void enforceGrading() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int l = L - 1; l >= 1; --l) {
                int n = lev[l].n;
                for (int k = 0; k < n; ++k)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            if (lev[l].at(i, j, k) != Cell::LEAF) continue;
                            for (int s = 0; s < 6; ++s) {
                                int ni = i + FDI[s], nj = j + FDJ[s], nk = k + FDK[s];
                                if (!lev[l].in(ni, nj, nk)) continue;
                                if (lev[l].at(ni, nj, nk) != Cell::OUTSIDE) continue;
                                int ii, jj, kk;
                                int al = coveringLeaf(l, ni, nj, nk, ii, jj, kk);
                                if (al >= 0 && al < l - 1) { splitLeaf(al, ii, jj, kk); changed = true; }
                            }
                        }
            }
        }
    }

    void numberDofs() {
        ndof = 0;
        dof_level.clear(); dof_i.clear(); dof_j.clear(); dof_k.clear();
        for (int l = 0; l < L; ++l) {
            lev[l].dof.assign(lev[l].type.size(), -1);
            int n = lev[l].n;
            for (int k = 0; k < n; ++k)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        if (lev[l].at(i, j, k) == Cell::LEAF) {
                            lev[l].dof[lev[l].idx(i, j, k)] = ndof;
                            dof_level.push_back(l); dof_i.push_back(i);
                            dof_j.push_back(j); dof_k.push_back(k);
                            ++ndof;
                        }
        }
    }
};

}  // namespace semistruct

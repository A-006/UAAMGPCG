// ─────────────────────────────────────────────────────────────────────────
// Semi-structured adaptive grid (2D) — reproduction of the data structure in
//   Wang, Sun, Zhu, "Matrix-Free Multigrid with Algebraically Consistent
//   Coarsening on Adaptive Octrees" (arXiv:2604.18886).
//
// "Semi-structured" = a stack of uniform level grids + a leaf/refined mask
// (SPGrid / tile style). Each level l is a uniform n_l × n_l grid on [0,1]^2
// with n_l = n0 * 2^l and cell size h_l = 1/n_l. A cell is one of:
//   OUTSIDE  — not part of the tree at this level (a coarser ancestor is the leaf)
//   LEAF     — an active degree of freedom (covers its area exactly once)
//   REFINED  — an "inner" cell, parent of finer cells (not a DOF itself)
// The union of LEAF cells across all levels tiles the domain exactly once.
//
// The grid is built top-down from a refinement predicate and then 2:1 balanced
// (graded): two face-adjacent leaves differ by at most one level. This module
// is self-contained (no dependency on the rest of the repo) and uses flat SoA
// arrays so it can later be mapped onto GPU tiles.
// ─────────────────────────────────────────────────────────────────────────
#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <cassert>
#include <array>

namespace semistruct {

enum class Cell : uint8_t { OUTSIDE = 0, LEAF = 1, REFINED = 2 };

struct Level {
    int n = 0;       // cells per side at this level
    double h = 0.0;  // cell size
    std::vector<Cell> type;  // size n*n, column-major (i + j*n)
    std::vector<int> dof;    // global DOF id for LEAF cells, else -1

    inline int idx(int i, int j) const { return i + j * n; }
    inline bool in(int i, int j) const { return i >= 0 && i < n && j >= 0 && j < n; }
    inline Cell at(int i, int j) const { return type[idx(i, j)]; }
};

// A face coupling produced by the composite operator: this leaf couples to the
// leaf `other` (global DOF id) with symmetric conservative coefficient `kappa`.
struct Coupling {
    int other;     // global DOF id of the neighbour leaf
    double kappa;  // symmetric flux coefficient (subface_len / center_dist)
};

// Boundary condition on the four domain sides.
enum class BC { Neumann, Dirichlet };

struct AdaptiveGrid2D {
    int L = 0;        // number of levels (0 = coarsest)
    int n0 = 0;       // base resolution at level 0
    std::vector<Level> lev;
    int ndof = 0;     // number of LEAF cells (global DOFs)
    // For each DOF: which (level,i,j) leaf it is.
    std::vector<int> dof_level, dof_i, dof_j;
    std::array<BC, 4> bc{BC::Neumann, BC::Neumann, BC::Neumann, BC::Neumann}; // -x,+x,-y,+y

    // refineFn(level, i, j, h, cx, cy) → true if cell (level,i,j) must be split
    // into finer children. Called during top-down construction.
    using RefineFn = std::function<bool(int, int, int, double, double, double)>;

    void build(int levels, int base_n, const RefineFn& refineFn) {
        L = levels;
        n0 = base_n;
        lev.assign(L, Level{});
        for (int l = 0; l < L; ++l) {
            int n = n0 << l;
            lev[l].n = n;
            lev[l].h = 1.0 / n;
            lev[l].type.assign((size_t)n * n, Cell::OUTSIDE);
        }
        for (int j = 0; j < n0; ++j)
            for (int i = 0; i < n0; ++i)
                mark(0, i, j, refineFn);
        enforceGrading(refineFn);
        numberDofs();
    }

    // Centre coordinate of cell (l,i,j).
    inline double cx(int l, int i) const { return (i + 0.5) * lev[l].h; }
    inline double cy(int l, int j) const { return (j + 0.5) * lev[l].h; }

    // The leaf that covers the centre of cell (l,i,j): walk up until a LEAF
    // ancestor is found. Returns level (and writes ii,jj). Returns -1 if the
    // queried cell is REFINED (covering leaf is finer — caller handles that).
    int coveringLeaf(int l, int i, int j, int& ii, int& jj) const {
        int cl = l, ci = i, cj = j;
        while (cl >= 0) {
            Cell c = lev[cl].at(ci, cj);
            if (c == Cell::LEAF) { ii = ci; jj = cj; return cl; }
            if (c == Cell::REFINED) return -1;  // finer covers it
            // OUTSIDE → go to parent
            cl -= 1; ci >>= 1; cj >>= 1;
        }
        return -1;
    }

    // Enumerate the composite-operator couplings for leaf DOF `d` plus boundary
    // terms. Appends to `out`; returns added diagonal from boundary Dirichlet
    // faces in `bc_diag`, and the Dirichlet RHS shift coefficient sum (face
    // kappa where the boundary value is gd) per side via callback handled by
    // caller (we expose Dirichlet faces through `dir_faces`).
    struct DirFace { double kappa; int side; };  // ghost value applied by caller
    void couplings(int d, std::vector<Coupling>& out, double& bc_diag,
                   std::vector<DirFace>& dir_faces) const {
        out.clear();
        dir_faces.clear();
        bc_diag = 0.0;
        int l = dof_level[d], i = dof_i[d], j = dof_j[d];
        double h = lev[l].h;
        // 4 faces: side 0:-x 1:+x 2:-y 3:+y
        const int di[4] = {-1, 1, 0, 0};
        const int dj[4] = {0, 0, -1, 1};
        for (int s = 0; s < 4; ++s) {
            int ni = i + di[s], nj = j + dj[s];
            if (!lev[l].in(ni, nj)) {
                // domain boundary
                if (bc[s] == BC::Dirichlet) {
                    // ghost at distance h/2; kappa = h /(h/2) = 2 (per unit length: subface=h, dist=h/2)
                    double kappa = 1.0 / 0.5;  // = 2 (subface h cancels with /h units → see below)
                    // In our nondimensional flux: kappa = subface_len/center_dist.
                    // subface_len = h, center_dist = h/2 → kappa = 2.
                    kappa = h / (h * 0.5);
                    bc_diag += kappa;
                    dir_faces.push_back({kappa, s});
                }
                // Neumann: no flux, nothing added.
                continue;
            }
            Cell nc = lev[l].at(ni, nj);
            if (nc == Cell::LEAF) {
                int nd = lev[l].dof[lev[l].idx(ni, nj)];
                out.push_back({nd, kappaSame()});
            } else if (nc == Cell::REFINED) {
                // this leaf is the COARSE side: couple to the two fine children
                // on the shared face at level l+1.
                addFineNeighbours(l, i, j, ni, nj, s, out);
            } else {  // OUTSIDE → coarse ancestor is the covering leaf (FINE side)
                int ii, jj;
                int al = coveringLeaf(l, ni, nj, ii, jj);
                assert(al >= 0 && al < l);
                int nd = lev[al].dof[lev[al].idx(ii, jj)];
                out.push_back({nd, kappaCoarseFine(h, lev[al].h)});
            }
        }
    }

    // The two children of refined cell (ni,nj) at level+1 that touch face `s`
    // of the COARSE cell on the opposite side. s: 0:-x 1:+x 2:-y 3:+y (coarse's face).
    static void childrenOnFace(int ni, int nj, int s, int cs[2][2]) {
        int bi = 2 * ni, bj = 2 * nj;
        switch (s) {
            case 0:  // coarse -x → neighbour +x column (bi+1)
                cs[0][0] = bi + 1; cs[0][1] = bj;
                cs[1][0] = bi + 1; cs[1][1] = bj + 1; break;
            case 1:  // coarse +x → neighbour -x column (bi)
                cs[0][0] = bi; cs[0][1] = bj;
                cs[1][0] = bi; cs[1][1] = bj + 1; break;
            case 2:  // coarse -y → neighbour +y row (bj+1)
                cs[0][0] = bi;     cs[0][1] = bj + 1;
                cs[1][0] = bi + 1; cs[1][1] = bj + 1; break;
            default: // coarse +y → neighbour -y row (bj)
                cs[0][0] = bi;     cs[0][1] = bj;
                cs[1][0] = bi + 1; cs[1][1] = bj; break;
        }
    }

    inline int dofAt(int l, int i, int j) const { return lev[l].dof[lev[l].idx(i, j)]; }

    // Same-level coupling coefficient (nondimensional): subface=h, dist=h → 1.
    static inline double kappaSame() { return 1.0; }
    // Coarse/fine: subface = smaller h (fine), dist = hf/2 + hc/2.
    static inline double kappaCoarseFine(double hf, double hc) {
        double sub = (hf < hc ? hf : hc);
        double dist = 0.5 * (hf + hc);
        return sub / dist;
    }

private:
    void mark(int l, int i, int j, const RefineFn& refineFn) {
        if (l == L - 1) { lev[l].type[lev[l].idx(i, j)] = Cell::LEAF; return; }
        double h = lev[l].h;
        if (refineFn(l, i, j, h, cx(l, i), cy(l, j))) {
            lev[l].type[lev[l].idx(i, j)] = Cell::REFINED;
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di)
                    mark(l + 1, 2 * i + di, 2 * j + dj, refineFn);
        } else {
            lev[l].type[lev[l].idx(i, j)] = Cell::LEAF;
        }
    }

    // Refine a leaf (l,i,j) into 4 children leaves (used by grading).
    void splitLeaf(int l, int i, int j) {
        lev[l].type[lev[l].idx(i, j)] = Cell::REFINED;
        for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di)
                lev[l + 1].type[lev[l + 1].idx(2 * i + di, 2 * j + dj)] = Cell::LEAF;
    }

    // 2:1 balance: if a leaf at level l is face-adjacent to a leaf at level < l-1,
    // refine that coarser leaf. Iterate to fixpoint.
    void enforceGrading(const RefineFn&) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int l = L - 1; l >= 1; --l) {
                int n = lev[l].n;
                const int di[4] = {-1, 1, 0, 0};
                const int dj[4] = {0, 0, -1, 1};
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        if (lev[l].at(i, j) != Cell::LEAF) continue;
                        for (int s = 0; s < 4; ++s) {
                            int ni = i + di[s], nj = j + dj[s];
                            if (!lev[l].in(ni, nj)) continue;
                            if (lev[l].at(ni, nj) != Cell::OUTSIDE) continue;
                            // covering leaf of neighbour is coarser; find its level
                            int ii, jj;
                            int al = coveringLeaf(l, ni, nj, ii, jj);
                            if (al >= 0 && al < l - 1) {
                                splitLeaf(al, ii, jj);  // refine coarse leaf one level
                                changed = true;
                            }
                        }
                    }
            }
        }
    }

    // Add couplings from coarse leaf (l,i,j) to the two fine children of the
    // refined neighbour (l, ni, nj) that touch the shared face s.
    void addFineNeighbours(int l, int /*i*/, int /*j*/, int ni, int nj, int s,
                           std::vector<Coupling>& out) const {
        // Children of (ni,nj) at level l+1 occupy [2ni,2ni+1]x[2nj,2nj+1].
        double hf = lev[l + 1].h, hc = lev[l].h;
        double kappa = kappaCoarseFine(hf, hc);
        // The two children touching face s of the coarse cell:
        // s=-x: the coarse cell's -x face → neighbour is to the -x; its children
        //       touching the face are the +x column of (ni,nj)'s children.
        int cs[2][2];  // up to 2 child coords
        childrenOnFace(ni, nj, s, cs);
        for (int t = 0; t < 2; ++t) {
            int fi = cs[t][0], fj = cs[t][1];
            int nd = lev[l + 1].dof[lev[l + 1].idx(fi, fj)];
            out.push_back({nd, kappa});
        }
    }

    void numberDofs() {
        ndof = 0;
        dof_level.clear(); dof_i.clear(); dof_j.clear();
        for (int l = 0; l < L; ++l) {
            lev[l].dof.assign(lev[l].type.size(), -1);
            int n = lev[l].n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (lev[l].at(i, j) == Cell::LEAF) {
                        lev[l].dof[lev[l].idx(i, j)] = ndof;
                        dof_level.push_back(l); dof_i.push_back(i); dof_j.push_back(j);
                        ++ndof;
                    }
        }
    }
};

}  // namespace semistruct
